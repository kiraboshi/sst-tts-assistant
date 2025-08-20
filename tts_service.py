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

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def get_engine(self):
        if not self._enabled:
            return None
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine

            if not self._speaker_wav or not os.path.exists(self._speaker_wav):
                print("[TTS] Disabled (missing --tts-speaker-wav or file not found)")
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
                        getattr(getattr(self._engine, "synthesizer", None), "output_sample_rate", 24000)
                    )
                except Exception:
                    self._sample_rate = 24000

                print(
                    f"[TTS] Loaded model '{self._model_id}' on {device} in {time.time()-t0:.1f}s (sr={self.sample_rate})"
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
        phrase = phrase or os.getenv("WARMUP_TTS_TEXT", "Initializing voice.")
        return self.synthesize(phrase)

    def play(self, audio: np.ndarray, blocking: bool = True) -> bool:
        try:
            import sounddevice as sd
        except Exception:
            print("[TTS] playback unavailable (sounddevice not installed)")
            return False
        try:
            sd.play(audio, samplerate=self.sample_rate, blocking=blocking)
            return True
        except Exception as e:
            print(f"[TTS] playback failed: {e}")
            return False


