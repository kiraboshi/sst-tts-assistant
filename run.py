import sys
import time
import re
import argparse
import os
import threading
import numpy as np

from voice_activation import SileroVoiceActivation, VoiceActivationConfig
from openai_client import OpenAIAssistant


def main() -> int:
    # Configure voice activation (mirrors parameters used in pipecat's Silero VAD)

    # CLI options for customizing start/stop phrases
    parser = argparse.ArgumentParser(description="Silero voice activation demo")
    parser.add_argument(
        "--start-words",
        "--start",
        dest="start_words",
        default="hello computer",
        help="Start phrase to enter listen mode",
    )
    parser.add_argument(
        "--stop-words",
        "--stop",
        dest="stop_words",
        default="thank you",
        help="Stop phrase to exit listen mode",
    )
    parser.add_argument(
        "--asr-model",
        dest="asr_model_id",
        default=os.getenv("ASR_MODEL_ID"),
        help=(
            "Hugging Face model id for ASR (e.g. openai/whisper-small). "
            "Overrides automatic default."
        ),
    )
    parser.add_argument(
        "--asr-local-only",
        action="store_true",
        help=(
            "Do not download models; use only local cache "
            "(TRANSFORMERS_OFFLINE-like)."
        ),
    )
    parser.add_argument(
        "--eager-asr",
        action="store_true",
        help="Load the ASR pipeline immediately at startup (disables lazy load).",
    )
    parser.add_argument(
        "--asr-warmup",
        action="store_true",
        help=(
            "Start loading the ASR pipeline in a background thread after startup."
        ),
    )
    # TTS options
    parser.add_argument(
        "--tts",
        dest="tts_enabled",
        action="store_true",
        help="Enable TTS for assistant responses (requires Coqui TTS and a speaker wav)",
    )
    parser.add_argument(
        "--no-tts",
        dest="tts_enabled",
        action="store_false",
        help="Disable TTS for assistant responses",
    )
    parser.set_defaults(tts_enabled=None)
    parser.add_argument(
        "--tts-speaker-wav",
        dest="tts_speaker_wav",
        default=(
            os.getenv("TTS_SPEAKER_WAV")
            or ("test.wav" if os.path.exists("test.wav") else None)
        ),
        help=(
            "Path to reference speaker audio for voice cloning (xtts_v2 requires this)."
        ),
    )
    parser.add_argument(
        "--tts-language",
        dest="tts_language",
        default=os.getenv("TTS_LANGUAGE", "en"),
        help="Target language code for TTS (e.g., 'en')",
    )
    parser.add_argument(
        "--tts-model-id",
        dest="tts_model_id",
        default=os.getenv(
            "TTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2"
        ),
        help="Coqui TTS model id to use",
    )
    parser.add_argument(
        "--tts-vocoder-id",
        dest="tts_vocoder_id",
        default=os.getenv(
            "TTS_VOCODER_ID",
            "vocoder_models/universal/libri-tts/wavegrad",
        ),
        help=(
            "Coqui TTS vocoder id/path to use (e.g. "
            "'vocoder_models/universal/libri-tts/wavegrad')"
        ),
    )
    args = parser.parse_args()

    # Feature flag: enable LLM-based fuzzy wake/sleep checks when truthy
    llm_checks_enabled = os.getenv("ASSISTANT_LLM_CHECKS", "").lower() in ("1", "true", "yes", "on")

    def compile_phrase_regex(phrase: str):
        """Build a robust regex that matches a phrase, ignoring punctuation and casing.

        - Normalizes the phrase by stripping punctuation and lowercasing tokens
        - Ignores punctuation in the input by allowing non-alphanumerics between letters
        """
        raw_tokens = [t for t in re.split(r"\s+", phrase.strip()) if t]
        # Strip punctuation and normalize case for tokens
        tokens = [re.sub(r"[^A-Za-z0-9]+", "", t.lower()) for t in raw_tokens]
        tokens = [t for t in tokens if t]
        if not tokens:
            # Compile a regex that never matches
            return re.compile(r"(?!x)x")
        # Allow any non-alphanumeric (including spaces/punctuation) between characters and tokens
        connector = r"[^A-Za-z0-9]*"

        def token_pattern(tok: str) -> str:
            # Allow punctuation between characters of the token
            chars = [re.escape(c) for c in tok]
            return connector.join(chars)

        parts = [token_pattern(tok) for tok in tokens]
        pattern = connector.join(parts)
        return re.compile(pattern, re.IGNORECASE)

    # Lazy ASR pipeline loader to avoid heavy startup cost
    asr_pipeline = None
    asr_lock = threading.Lock()

    def get_asr_pipeline():
        nonlocal asr_pipeline
        if asr_pipeline is not None:
            return asr_pipeline
        with asr_lock:
            if asr_pipeline is not None:
                return asr_pipeline
            # Import heavy deps only when needed
            import torch as _torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline as hf_pipeline,
            )

            device_str = "cuda:0" if _torch.cuda.is_available() else "cpu"
            torch_dtype = (
                _torch.float16 if _torch.cuda.is_available() else _torch.float32
            )
            default_cpu_model = os.getenv(
                "ASR_MODEL_ID_CPU", "openai/whisper-small"
            )
            default_gpu_model = os.getenv(
                "ASR_MODEL_ID_GPU", "openai/whisper-large-v3-turbo"
            )
            model_id = (
                args.asr_model_id
                or (
                    default_gpu_model
                    if _torch.cuda.is_available()
                    else default_cpu_model
                )
            )
            local_only = bool(
                args.asr_local_only
                or os.getenv("ASR_LOCAL_ONLY") in ("1", "true", "yes", "on")
            )

            t0 = time.time()
            print(
                f"[ASR] Loading model '{model_id}' on {device_str} "
                f"(dtype={torch_dtype}) …"
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=local_only,
            )
            model.to(device_str)
            processor = AutoProcessor.from_pretrained(
                model_id, local_files_only=local_only
            )
            asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_str,
            )
            print(f"[ASR] Loaded in {time.time() - t0:.1f}s")
            return asr_pipeline

    # Resolve TTS enablement and lazy loader
    if args.tts_enabled is None:
        env_tts = os.getenv("TTS_ENABLED", "").lower() in ("1", "true", "yes", "on")
        tts_enabled = env_tts if os.getenv("TTS_ENABLED") is not None else bool(args.tts_speaker_wav)
    else:
        tts_enabled = bool(args.tts_enabled)

    tts_engine = None
    tts_lock = threading.Lock()
    tts_sample_rate: int | None = None

    # Pause VAD briefly after TTS finishes to avoid immediate re-trigger
    vad_pause_after_tts_ms = int(os.getenv("TTS_VAD_DELAY_MS", "500"))

    def get_tts_engine():
        nonlocal tts_engine, tts_sample_rate
        if not tts_enabled:
            return None
        if tts_engine is not None:
            return tts_engine
        with tts_lock:
            if tts_engine is not None:
                return tts_engine
            # Validate speaker wav
            if not args.tts_speaker_wav or not os.path.exists(args.tts_speaker_wav):
                print("[TTS] Disabled (missing --tts-speaker-wav or file not found)")
                return None
            try:
                import torch as _torch
                import importlib as _importlib
                _tts_module = _importlib.import_module("TTS.api")
                _TTS = getattr(_tts_module, "TTS")
                # PyTorch 2.6+ defaults weights_only=True; allowlist XTTS config for safe unpickling
                try:
                    cfg_mod = _importlib.import_module("TTS.tts.configs.xtts_config")
                    XttsConfig = getattr(cfg_mod, "XttsConfig", None)
                    if XttsConfig is not None:
                        try:
                            from torch.serialization import add_safe_globals as _add_safe_globals  # type: ignore
                        except Exception:
                            _add_safe_globals = None  # type: ignore
                        if _add_safe_globals is not None:
                            _add_safe_globals([XttsConfig])
                        try:
                            from torch.serialization import safe_globals as _safe_globals  # type: ignore
                        except Exception:
                            _safe_globals = None  # type: ignore
                    else:
                        _safe_globals = None  # type: ignore
                except Exception:
                    # Best-effort; continue to try loading the model
                    XttsConfig = None
                    _safe_globals = None  # type: ignore
            except Exception as e:
                print(
                    f"[TTS] Disabled (package not available: "
                    f"{type(e).__name__}: {e})"
                )
                return None
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            t0 = time.time()
            try:
                # Use safe_globals context if available for secure allowlist
                try:
                    from contextlib import nullcontext as _nullcontext
                except Exception:
                    _nullcontext = None  # type: ignore
                ctx = (
                    _safe_globals([XttsConfig])  # type: ignore
                    if ("_safe_globals" in locals() and _safe_globals and XttsConfig)
                    else (_nullcontext() if _nullcontext else None)
                )
                
                def _attempt_init():
                    nonlocal tts_engine
                    if ctx is not None:
                        with ctx:  # type: ignore
                            if getattr(args, "tts_vocoder_id", None):
                                tts_engine = _TTS(
                                    model_name=args.tts_model_id,
                                    vocoder_name=args.tts_vocoder_id,
                                ).to(device)
                            else:
                                tts_engine = _TTS(
                                    args.tts_model_id
                                ).to(device)
                    else:
                        if getattr(args, "tts_vocoder_id", None):
                            tts_engine = _TTS(
                                args.tts_model_id,
                                args.tts_vocoder_id,
                            ).to(device)
                        else:
                            tts_engine = _TTS(args.tts_model_id).to(device)

                try:
                    _attempt_init()
                except Exception as e:
                    # Dynamic allowlist: parse unsupported global and retry once
                    msg = str(e)
                    if "Unsupported global: GLOBAL" in msg:
                        try:
                            import re as _re
                            m = _re.search(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)", msg)
                            if m:
                                full_name = m.group(1)
                                mod_name, _, cls_name = full_name.rpartition(".")
                                if mod_name and cls_name:
                                    try:
                                        mod = _importlib.import_module(mod_name)
                                        cls = getattr(mod, cls_name)
                                        try:
                                            from torch.serialization import add_safe_globals as _add_safe_globals  # type: ignore
                                        except Exception:
                                            _add_safe_globals = None  # type: ignore
                                        if _add_safe_globals is not None:
                                            _add_safe_globals([cls])
                                            _attempt_init()
                                        else:
                                            raise e
                                    except Exception:
                                        raise e
                            else:
                                raise e
                        except Exception as e2:
                            print(f"[TTS] init failed: {type(e2).__name__}: {e2}")
                            tts_engine = None
                            return None
                    else:
                        print(f"[TTS] init failed: {type(e).__name__}: {e}")
                        tts_engine = None
                        return None
            except Exception as e:
                print(f"[TTS] init failed (outer): {type(e).__name__}: {e}")
                tts_engine = None
                return None
            except Exception as e:
                print(f"[TTS] init failed (outer): {type(e).__name__}: {e}")
                tts_engine = None
                return None
            # Try to detect model output sample rate; fall back to 24000
            try:
                tts_sample_rate = getattr(
                    getattr(tts_engine, "synthesizer", None),
                    "output_sample_rate",
                    None,
                )
            except Exception:
                tts_sample_rate = None
            if not tts_sample_rate:
                tts_sample_rate = 24000
            print(
                "[TTS] Loaded model '" + args.tts_model_id +
                f"' on {device} in {time.time() - t0:.1f}s "
                f"(sr={tts_sample_rate})"
            )
            return tts_engine

    def speak_text(text: str) -> bool:
        nonlocal vad_pause_after_tts_ms
        if not text:
            return False
        engine = get_tts_engine()
        if engine is None:
            return False
        try:
            wav = engine.tts(
                text=text,
                speaker_wav=args.tts_speaker_wav,
                # language=args.tts_language,
                pitch_scale=1.2,
            )
            wav_np = np.asarray(wav, dtype=np.float32)
            try:
                sd.play(
                    wav_np,
                    samplerate=int(tts_sample_rate or 24000),
                    blocking=True,
                )
            except Exception as e:
                print(f"[TTS] playback failed: {e}")
                return False
            # Simple cooldown: sleep to allow the room to quiet down
            try:
                time.sleep(max(0.0, vad_pause_after_tts_ms / 1000.0))
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[TTS] synthesis failed: {type(e).__name__}: {e}")
            return False

    def warmup_models() -> None:
        """Warm up TTS (XTTS) and ASR (Whisper) by first speaking a short phrase
        and then running a dummy transcription. This avoids first-use latency.
        """
        wav_np: np.ndarray | None = None
        # TTS warm-up
        try:
            if tts_enabled:
                engine = get_tts_engine()
                if engine is not None:
                    phrase = os.getenv("WARMUP_TTS_TEXT", "Initializing voice.")
                    wav = engine.tts(
                        text=phrase,
                        speaker_wav=args.tts_speaker_wav,
                        # language=args.tts_language,
                    )
                    wav_np = np.asarray(wav, dtype=np.float32)

                    try:
                        sd.play(
                            wav_np,
                            samplerate=int(tts_sample_rate or 24000),
                            blocking=True,
                        )
                    except Exception as e:
                        print(f"[WARMUP] TTS playback failed: {e}")
        except Exception as e:
            print(f"[WARMUP] TTS warm-up failed: {e}")

        # ASR warm-up
        try:
            pipe = get_asr_pipeline()
            if pipe is not None:
                try:
                    sr_tgt = config.sample_rate
                    if wav_np is not None and wav_np.size > 0:
                        # Resample TTS audio to ASR sample rate (default 16k)
                        # using linear interpolation
                        sr_src = int(tts_sample_rate or 24000)
                        if sr_src != sr_tgt:
                            src_idx = np.arange(
                                wav_np.shape[0],
                                dtype=np.float64,
                            )
                            new_len = max(
                                1,
                                int(
                                    round(
                                        wav_np.shape[0] * (sr_tgt / sr_src)
                                    )
                                ),
                            )
                            dst_idx = np.linspace(
                                0.0,
                                float(wav_np.shape[0] - 1),
                                new_len,
                                dtype=np.float64,
                            )
                            wav_for_asr = np.interp(
                                dst_idx,
                                src_idx,
                                wav_np,
                            ).astype(np.float32)
                        else:
                            wav_for_asr = wav_np.astype(np.float32)
                    else:
                        # 0.5s of silence if no TTS audio
                        wav_for_asr = np.zeros(int(sr_tgt * 0.5), dtype=np.float32)

                    _ = pipe({"array": wav_for_asr, "sampling_rate": sr_tgt})
                except Exception as e:
                    print(f"[WARMUP] ASR inference failed: {e}")
        except Exception as e:
            print(f"[WARMUP] ASR warm-up failed: {e}")

    config = VoiceActivationConfig(
        sample_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        frame_ms=32,
    )
    t_vad0 = time.time()
    activation = SileroVoiceActivation(config)
    print(f"[VAD] Initialized in {time.time() - t_vad0:.2f}s")

    # Optional eager load or background warm-up
    if args.eager_asr:
        _ = get_asr_pipeline()
    elif args.asr_warmup:
        threading.Thread(target=get_asr_pipeline, daemon=True).start()

    # OpenAI assistant wrapper
    assistant = OpenAIAssistant()

    # Try to use the microphone for a simple realtime demo (requires sounddevice)
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Install with: pip install sounddevice")
        return 1

    # Precompute short tones for listen ON/OFF notifications
    def _generate_beep(frequency_hz: float, duration_ms: int = 180, volume: float = 0.2) -> np.ndarray:
        t = np.linspace(0.0, duration_ms / 1000.0, int(config.sample_rate * (duration_ms / 1000.0)), endpoint=False, dtype=np.float32)
        return (volume * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)

    beep_on: np.ndarray = _generate_beep(1000.0, duration_ms=200, volume=0.25)
    beep_off: np.ndarray = _generate_beep(440.0, duration_ms=200, volume=0.25)
    beep_ready: np.ndarray = _generate_beep(800.0, duration_ms=180, volume=0.22)
    beep_reset: np.ndarray = _generate_beep(600.0, duration_ms=180, volume=0.22)

    def _play_beep(buffer: np.ndarray) -> None:
        try:
            sd.play(buffer, samplerate=config.sample_rate, blocking=False)
        except Exception as e:
            print(f"[AUDIO] beep failed: {e}")

    # Speak once and run a dummy transcription so XTTS and Whisper are ready
    print("[WARMUP] Starting warm-up …")
    warmup_models()
    print("[WARMUP] Done.")

    frame_samples = int(config.sample_rate * (config.frame_ms / 1000.0))
    audio_buffer: list[np.ndarray] = []
    is_recording_segment = False

    # Simple command-controlled transcription state
    MODE_IDLE = "idle"
    MODE_LISTEN = "listen"
    mode = MODE_IDLE
    transcript_parts: list[str] = []

    # Command detection regex (robust to punctuation between words)
    wake_re = compile_phrase_regex(args.start_words)
    stop_re = compile_phrase_regex(args.stop_words)
    reset_re = compile_phrase_regex("reset")

    # type: ignore[override]xx
    def audio_callback(indata: np.ndarray, _frames: int, _time_info, status):
        nonlocal is_recording_segment
        if status:
            # Print non-fatal audio driver warnings
            print(status, file=sys.stderr)

        # Resample or slice to expected frame size if host provides different blocksize
        mono = indata[:, 0] if indata.ndim > 1 else indata
        if mono.shape[0] != frame_samples:
            # Adjust to the configured frame size
            if mono.shape[0] > frame_samples:
                mono = mono[:frame_samples]
            else:
                pad = np.zeros(frame_samples - mono.shape[0], dtype=mono.dtype)
                mono = np.concatenate([mono, pad])

        event = activation.process_frame(mono.astype(np.float32))

        if activation.is_active and not is_recording_segment:
            # Begin accumulating the active speech segment
            is_recording_segment = True
            audio_buffer.clear()
        if is_recording_segment:
            audio_buffer.append(np.copy(mono))

        if event == "speech_start":
            print("[VAD] speech_start")
        elif event == "speech_end":
            print("[VAD] speech_end")
            is_recording_segment = False
            # On segment end, we have a full utterance in audio_buffer
            segment = np.concatenate(audio_buffer) if audio_buffer else None
            if segment is not None:
                on_utterance_finished(segment, config.sample_rate)
            audio_buffer.clear()

    def on_utterance_finished(segment: np.ndarray, _sample_rate: int) -> None:
        """Handle end of a VAD speech segment: wake/stop control and transcription."""
        nonlocal mode, transcript_parts

        # STT inference for this segment using Whisper (lazy-loaded)
        try:
            pipe = get_asr_pipeline()
            result = pipe(
                {
                    "array": segment.astype(np.float32),
                    "sampling_rate": config.sample_rate,
                }
            )
            text = (
                result.get("text", "") if isinstance(result, dict) else str(result)
            ).strip()
        except Exception as e:
            print(f"[ASR] transcription failed: {e}")
            text = ""

        if not text:
            return

        # Idle: look for wake command
        if mode == MODE_IDLE:
            print(f"Idle: {text}")
            m = wake_re.search(text)
            should_wake = False
            if m:
                should_wake = True
                trailing_after_regex = text[m.end():].strip()
            else:
                # If regex did not match, optionally ask the LLM for a fuzzy wake check
                trailing_after_regex = ""
                if assistant.enabled and llm_checks_enabled:
                    should_wake = assistant.check_wake(text)

            if should_wake:
                mode = MODE_LISTEN
                transcript_parts.clear()
                if assistant.enabled:
                    assistant.reset_conversation()
                if trailing_after_regex:
                    transcript_parts.append(trailing_after_regex)
                print(f">>> Listen mode ON. Say '{args.stop_words}' to finish.")
                _play_beep(beep_on)
            return

        # Listen: handle reset/stop; otherwise accumulate
        # Reset command: clears current transcript and assistant history, stays in listen mode
        m_reset = reset_re.search(text)
        if m_reset:
            transcript_parts.clear()
            if assistant.enabled:
                assistant.reset_conversation()
            print(">>> Conversation reset.")
            _play_beep(beep_reset)
            return

        # Stop command detection
        m_stop = stop_re.search(text)
        should_sleep = False
        leading = None
        if m_stop:
            should_sleep = True
            leading = text[:m_stop.start()].strip()
        else:
            # If regex did not match, optionally ask the LLM for a fuzzy stop check
            if assistant.enabled and llm_checks_enabled:
                should_sleep = assistant.check_sleep(text, args.stop_words)

        if should_sleep:
            if leading:
                transcript_parts.append(leading)
            final_text = " ".join(p for p in (part.strip() for part in transcript_parts) if p)
            if final_text:
                print(final_text)
            else:
                print("")
            transcript_parts.clear()
            mode = MODE_IDLE
            print(">>> Listen mode OFF.")
            _play_beep(beep_off)
            return

        print(f"Text: {text}")

        if assistant.enabled:
            response = assistant.chat_completion([text], system_text="You are a helpful assistant. Answer the user query in one sentence.")
            if response:
                print(f"[OPENAI] {response}")
                # Optionally speak the assistant response
                if tts_enabled:
                    _ = speak_text(response)

        transcript_parts.append(text)

    print(f"Listening from default microphone. Say '{args.start_words}' to start, '{args.stop_words}' to finish. Press Ctrl+C to exit.")
    _play_beep(beep_ready)
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
