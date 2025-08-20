import sys
import time
import argparse
import os
import threading
import numpy as np

from voice_activation import SileroVoiceActivation, VoiceActivationConfig
from openai_client import OpenAIAssistant
from asr_service import ASRService
from tts_service import TTSService
from audio_utils import generate_beep
from warmup import warmup_models
from controller import ConversationController


def main() -> int:
    parser = argparse.ArgumentParser(description="Silero voice activation demo")
    parser.add_argument("--start-words", "--start", dest="start_words", default="hello computer", help="Start phrase to enter listen mode")
    parser.add_argument("--stop-words", "--stop", dest="stop_words", default="thank you", help="Stop phrase to exit listen mode")
    parser.add_argument("--asr-model", dest="asr_model_id", default=os.getenv("ASR_MODEL_ID"), help="Hugging Face model id for ASR (e.g. openai/whisper-small). Overrides automatic default.")
    parser.add_argument("--asr-local-only", action="store_true", help="Do not download models; use only local cache (TRANSFORMERS_OFFLINE-like).")
    parser.add_argument("--eager-asr", action="store_true", help="Load the ASR pipeline immediately at startup (disables lazy load).")
    parser.add_argument("--asr-warmup", action="store_true", help="Start loading the ASR pipeline in a background thread after startup.")
    parser.add_argument("--tts", dest="tts_enabled", action="store_true", help="Enable TTS for assistant responses (requires Coqui TTS and a speaker wav)")
    parser.add_argument("--no-tts", dest="tts_enabled", action="store_false", help="Disable TTS for assistant responses")
    parser.set_defaults(tts_enabled=None)
    parser.add_argument("--tts-speaker-wav", dest="tts_speaker_wav", default=(os.getenv("TTS_SPEAKER_WAV") or ("test.wav" if os.path.exists("test.wav") else None)), help="Path to reference speaker audio for voice cloning (xtts_v2 requires this).")
    parser.add_argument("--tts-language", dest="tts_language", default=os.getenv("TTS_LANGUAGE", "en"), help="Target language code for TTS (e.g., 'en')")
    parser.add_argument("--tts-model-id", dest="tts_model_id", default=os.getenv("TTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2"), help="Coqui TTS model id to use")
    parser.add_argument("--tts-vocoder-id", dest="tts_vocoder_id", default=os.getenv("TTS_VOCODER_ID", "vocoder_models/universal/libri-tts/wavegrad"), help="Coqui TTS vocoder id/path to use")
    args = parser.parse_args()

    llm_checks_enabled = os.getenv("ASSISTANT_LLM_CHECKS", "").lower() in ("1", "true", "yes", "on")

    # Resolve TTS enablement
    if args.tts_enabled is None:
        env_tts = os.getenv("TTS_ENABLED", "").lower() in ("1", "true", "yes", "on")
        tts_enabled = env_tts if os.getenv("TTS_ENABLED") is not None else bool(args.tts_speaker_wav)
    else:
        tts_enabled = bool(args.tts_enabled)

    # Services
    asr = ASRService(model_id=args.asr_model_id, local_only=bool(args.asr_local_only or os.getenv("ASR_LOCAL_ONLY") in ("1", "true", "yes", "on")), eager_load=bool(args.eager_asr))
    tts = TTSService(enabled=tts_enabled, speaker_wav=args.tts_speaker_wav, model_id=args.tts_model_id, vocoder_id=args.tts_vocoder_id)

    if args.asr_warmup:
        threading.Thread(target=asr.get_pipeline, daemon=True).start()

    assistant = OpenAIAssistant()

    # Mic access
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Install with: pip install sounddevice")
        return 1

    config = VoiceActivationConfig(sample_rate=16000, threshold=0.5, min_silence_duration_ms=300, speech_pad_ms=100, frame_ms=32)
    t_vad0 = time.time()
    activation = SileroVoiceActivation(config)
    print(f"[VAD] Initialized in {time.time() - t_vad0:.2f}s")

    # Beeps
    beep_on = generate_beep(1000.0, duration_ms=200, sample_rate=config.sample_rate, volume=0.25)
    beep_off = generate_beep(440.0, duration_ms=200, sample_rate=config.sample_rate, volume=0.25)
    beep_ready = generate_beep(800.0, duration_ms=180, sample_rate=config.sample_rate, volume=0.22)
    beep_reset = generate_beep(600.0, duration_ms=180, sample_rate=config.sample_rate, volume=0.22)

    def play_beep(buf: np.ndarray) -> None:
        try:
            sd.play(buf, samplerate=config.sample_rate, blocking=False)
        except Exception as e:
            print(f"[AUDIO] beep failed: {e}")

    # TTS speak wrapper with brief VAD cooldown
    vad_pause_after_tts_ms = int(os.getenv("TTS_VAD_DELAY_MS", "500"))

    def speak_text(text: str) -> bool:
        if not text:
            return False
        wav = tts.synthesize(text)
        if wav is None:
            return False
        ok = tts.play(wav, blocking=True)
        try:
            time.sleep(max(0.0, vad_pause_after_tts_ms / 1000.0))
        except Exception:
            pass
        return ok

    # Allow controller to optionally speak via assistant.speak
    setattr(assistant, "speak", speak_text)

    # Warmup
    print("[WARMUP] Starting warm-up …")
    warmup_models(tts, asr, config.sample_rate)
    print("[WARMUP] Done.")

    # Controller
    controller = ConversationController(
        start_phrase=args.start_words,
        stop_phrase=args.stop_words,
        sample_rate=config.sample_rate,
        asr_transcribe=asr.transcribe,
        assistant=assistant,
        on_beep_on=lambda: play_beep(beep_on),
        on_beep_off=lambda: play_beep(beep_off),
        on_beep_reset=lambda: play_beep(beep_reset),
        llm_checks_enabled=llm_checks_enabled,
    )

    frame_samples = int(config.sample_rate * (config.frame_ms / 1000.0))
    audio_buffer: list[np.ndarray] = []
    is_recording_segment = False

    def audio_callback(indata: np.ndarray, _frames: int, _time_info, status):
        nonlocal is_recording_segment
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        if mono.shape[0] != frame_samples:
            if mono.shape[0] > frame_samples:
                mono = mono[:frame_samples]
            else:
                pad = np.zeros(frame_samples - mono.shape[0], dtype=mono.dtype)
                mono = np.concatenate([mono, pad])

        event = activation.process_frame(mono.astype(np.float32))

        if activation.is_active and not is_recording_segment:
            is_recording_segment = True
            audio_buffer.clear()
        if is_recording_segment:
            audio_buffer.append(np.copy(mono))

        if event == "speech_start":
            print("[VAD] speech_start")
        elif event == "speech_end":
            print("[VAD] speech_end")
            is_recording_segment = False
            segment = np.concatenate(audio_buffer) if audio_buffer else None
            if segment is not None:
                controller.handle_segment(segment)
            audio_buffer.clear()

    print(f"Listening from default microphone. Say '{args.start_words}' to start, '{args.stop_words}' to finish. Press Ctrl+C to exit.")
    play_beep(beep_ready)
    try:
        with sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        activation.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
