import numpy as np


def generate_beep(frequency_hz: float, duration_ms: int, sample_rate: int, volume: float = 0.2) -> np.ndarray:
    t = np.linspace(0.0, duration_ms / 1000.0, int(sample_rate * (duration_ms / 1000.0)), endpoint=False, dtype=np.float32)
    return (volume * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def linear_resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    if audio.size == 0:
        return audio.astype(np.float32)
    src_idx = np.arange(audio.shape[0], dtype=np.float64)
    new_len = max(1, int(round(audio.shape[0] * (dst_rate / src_rate))))
    dst_idx = np.linspace(0.0, float(audio.shape[0] - 1), new_len, dtype=np.float64)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


