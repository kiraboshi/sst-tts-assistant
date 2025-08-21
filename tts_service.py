import os
import time
import threading
from typing import Optional

import numpy as np


class TTSService:
    """
    Lazy-loading Coqui TTS (XTTS v2) service.

    - Loads only when first used
    - Returns generated waveform as numpy float32 array
    - Provides playback helper via sounddevice if available
    """

    def __init__(
        self,
        enabled: bool,
        speaker_wav: Optional[str] = None,
        model_id: Optional[str] = None,
        vocoder_id: Optional[str] = None,
        apply_ring_mod: Optional[bool] = None,
        ring_mod_freq_hz: Optional[float] = None,
        ring_mod_mix: Optional[float] = None,
        apply_pitch_shift: Optional[bool] = None,
        pitch_semitones: Optional[float] = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._speaker_wav = speaker_wav
        self._model_id = model_id or os.getenv(
            "TTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self._vocoder_id = vocoder_id or os.getenv(
            "TTS_VOCODER_ID", "vocoder_models/universal/libri-tts/wavegrad"
        )
        self._engine = None
        self._lock = threading.Lock()
        self._sample_rate: Optional[int] = None

        # Output processing configuration
        truthy = ("1", "true", "yes", "on")
        env_ring_mod = os.getenv("TTS_RING_MOD")
        if apply_ring_mod is None:
            # Default to ON if not explicitly configured
            self._apply_ring_mod: bool = (
                True
                if env_ring_mod is None
                else (str(env_ring_mod).lower() in truthy)
            )
        else:
            self._apply_ring_mod = bool(apply_ring_mod)

        try:
            env_freq = float(os.getenv("TTS_RING_FREQ", "90"))
        except Exception:
            env_freq = 90.0
        self._ring_mod_freq_hz: float = (
            env_freq if ring_mod_freq_hz is None else float(ring_mod_freq_hz)
        )

        # Ring modulation dry/wet mix (0..1). Lower mix reduces artifacts.
        try:
            env_mix = float(os.getenv("TTS_RING_MIX", "0.25"))
        except Exception:
            env_mix = 0.25
        mix_val = env_mix if ring_mod_mix is None else float(ring_mod_mix)
        # Clamp to [0, 1]
        if not (mix_val >= 0.0):
            mix_val = 0.0
        if not (mix_val <= 1.0):
            mix_val = 1.0
        self._ring_mod_mix: float = mix_val

        # Pitch shift configuration (formant-preserving via WORLD)
        truthy = ("1", "true", "yes", "on")
        env_pitch_on = os.getenv("TTS_PITCH_SHIFT")
        if apply_pitch_shift is None:
            self._apply_pitch_shift: bool = (
                False
                if env_pitch_on is None
                else (str(env_pitch_on).lower() in truthy)
            )
        else:
            self._apply_pitch_shift = bool(apply_pitch_shift)
        try:
            env_semi = float(os.getenv("TTS_PITCH_SEMITONES", "0"))
        except Exception:
            env_semi = 0.0
        self._pitch_semitones: float = (
            env_semi if pitch_semitones is None else float(pitch_semitones)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    @staticmethod
    def _ring_modulation(
        audio: np.ndarray, sr: int, mod_freq: float = 90.0, mix: float = 1.0
    ) -> np.ndarray:
        """
        Ring modulation with dry/wet mix.

        - Multiply signal with sine at mod_freq (Hz)
        - Blend with original using mix in [0, 1]
          (0 = dry/original, 1 = fully ring-modulated)
        """
        if audio.size == 0:
            return audio
        # Ensure mix is in [0, 1]
        if mix < 0.0:
            mix = 0.0
        elif mix > 1.0:
            mix = 1.0
        t_idx = np.arange(audio.shape[0], dtype=np.float32)
        t = t_idx / float(sr)
        modulator = np.sin(
            2.0 * np.pi * float(mod_freq) * t
        ).astype(np.float32)
        audio32 = audio.astype(np.float32, copy=False)
        if audio.ndim > 1:
            ringed = (audio32 * modulator[:, None]).astype(np.float32)
            if mix >= 1.0:
                return ringed
            if mix <= 0.0:
                return audio32
            return ((1.0 - mix) * audio32 + mix * ringed).astype(np.float32)
        ringed = (audio32 * modulator).astype(np.float32)
        if mix >= 1.0:
            return ringed
        if mix <= 0.0:
            return audio32
        return ((1.0 - mix) * audio32 + mix * ringed).astype(np.float32)

    def _process_output_audio(self, audio: np.ndarray) -> np.ndarray:
        processed = audio.astype(np.float32, copy=False)
        if self._apply_pitch_shift and abs(self._pitch_semitones) > 1e-6:
            processed = self._formant_preserving_pitch_shift(
                processed,
                self.sample_rate,
                self._pitch_semitones,
            )
        if self._apply_ring_mod and self._ring_mod_mix > 0.0:
            processed = self._ring_modulation(
                processed, self.sample_rate, self._ring_mod_freq_hz, self._ring_mod_mix
            )
        return processed

    @staticmethod
    def _formant_preserving_pitch_shift(
        audio: np.ndarray, sr: int, semitones: float
    ) -> np.ndarray:
        """
        Shift pitch by 'semitones' while preserving formants using
        WORLD vocoder.

        If pyworld is not available, returns the input unchanged.
        Accepts mono (N,) or multichannel (N, C) float arrays;
        returns float32.
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        try:
            import importlib
            pw = importlib.import_module("pyworld")
        except Exception:
            # pyworld not installed; skip
            print("[TTS] pyworld not installed; skipping pitch shift")
            return audio

        def _shift_channel(x: np.ndarray) -> np.ndarray:
            x64 = x.astype(np.float64, copy=False)
            frame_period_ms = 5.0
            # F0 extraction (robust)
            f0, t = pw.harvest(x64, sr)
            sp = pw.cheaptrick(x64, f0, t, sr)
            ap = pw.d4c(x64, f0, t, sr)
            factor = float(2.0 ** (semitones / 12.0))
            f0_new = f0 * factor
            y = pw.synthesize(f0_new, sp, ap, sr, frame_period=frame_period_ms)
            y32 = y.astype(np.float32)
            # Match original length by trimming or padding
            if y32.shape[0] > x.shape[0]:
                return y32[: x.shape[0]]
            if y32.shape[0] < x.shape[0]:
                pad = np.zeros(x.shape[0] - y32.shape[0], dtype=np.float32)
                return np.concatenate([y32, pad])
            return y32

        if audio.ndim == 1:
            return _shift_channel(audio)
        # Multichannel
        channels = []
        for c in range(audio.shape[1]):
            channels.append(_shift_channel(audio[:, c]))
        return np.stack(channels, axis=1)

    def get_engine(self):
        if not self._enabled:
            return None
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine

            if not self._speaker_wav or not os.path.exists(self._speaker_wav):
                print(
                    "[TTS] Disabled (missing --tts-speaker-wav or file not found)"
                )
                self._enabled = False
                return None
            try:
                import importlib
                torch = importlib.import_module("torch")
                TTS_api = importlib.import_module("TTS.api")
                TTS = getattr(TTS_api, "TTS")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                t0 = time.time()
                if self._vocoder_id:
                    self._engine = TTS(
                        model_name=self._model_id, vocoder_name=self._vocoder_id
                    ).to(device)
                else:
                    self._engine = TTS(self._model_id).to(device)

                # Detect output sample rate
                try:
                    self._sample_rate = int(
                        getattr(
                            getattr(self._engine, "synthesizer", None),
                            "output_sample_rate",
                            24000,
                        )
                    )
                except Exception:
                    self._sample_rate = 24000

                print(
                    f"[TTS] Loaded model '{self._model_id}' on {device} in "
                    f"{time.time() - t0:.1f}s (sr={self.sample_rate})"
                )
                return self._engine
            except Exception as e:
                print(f"[TTS] init failed: {type(e).__name__}: {e}")
                self._engine = None
                self._enabled = False
                return None

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        if not text or not self._enabled:
            return None
        engine = self.get_engine()
        if engine is None:
            return None
        try:
            wav = engine.tts(text=text, speaker_wav=self._speaker_wav)
            return np.asarray(wav, dtype=np.float32)
        except Exception as e:
            print(f"[TTS] synthesis failed: {type(e).__name__}: {e}")
            return None

    def warmup(self, phrase: Optional[str] = None) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        phrase = phrase or os.getenv("WARMUP_TTS_TEXT", "Initializing vector assistant system. Please wait.")
        return self.synthesize(phrase)

    def play(self, audio: np.ndarray, blocking: bool = True) -> bool:
        try:
            import importlib
            sd = importlib.import_module("sounddevice")
        except Exception:
            print("[TTS] playback unavailable (sounddevice not installed)")
            return False
        try:
            processed = self._process_output_audio(audio)
            sd.play(
                processed,
                samplerate=self.sample_rate,
                blocking=blocking,
            )
            return True
        except Exception as e:
            print(f"[TTS] playback failed: {e}")
            return False
