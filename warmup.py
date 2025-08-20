from typing import Optional

import numpy as np

from audio_utils import linear_resample


def warmup_models(tts, asr, target_sample_rate: int) -> None:
    """
    Warm up TTS (XTTS) and ASR (Whisper) by first speaking a short phrase
    and then running a dummy transcription of the audio (resampled to ASR rate).
    """
    wav_np: Optional[np.ndarray] = None

    # TTS warm-up
    try:
        if tts and getattr(tts, "enabled", False):
            wav_np = tts.warmup()
            if isinstance(wav_np, np.ndarray) and wav_np.size > 0:
                try:
                    tts.play(wav_np, blocking=True)
                except Exception as e:
                    print(f"[WARMUP] TTS playback failed: {e}")
    except Exception as e:
        print(f"[WARMUP] TTS warm-up failed: {e}")

    # ASR warm-up
    try:
        if hasattr(asr, "warmup"):
            if wav_np is not None and wav_np.size > 0:
                wav_for_asr = linear_resample(
                    wav_np.astype(np.float32), getattr(tts, "sample_rate", 24000), target_sample_rate
                )
            else:
                wav_for_asr = np.zeros(int(target_sample_rate * 0.5), dtype=np.float32)
            asr.warmup(wav_for_asr, target_sample_rate)
    except Exception as e:
        print(f"[WARMUP] ASR warm-up failed: {e}")


