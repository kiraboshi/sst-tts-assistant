import os
import time
import threading
from typing import Optional

import numpy as np


class ASRService:
    """
    Lazy-loading ASR service backed by Hugging Face Transformers Whisper pipeline.

    - Selects CPU/GPU model defaults based on torch.cuda availability
    - Respects local-only mode to avoid downloads
    - Provides simple transcribe() and warmup() helpers
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        local_only: bool = False,
        eager_load: bool = False,
    ) -> None:
        self._pipeline = None
        self._lock = threading.Lock()
        self._local_only = bool(local_only)

        # Defer torch import until needed for device selection
        import importlib

        self._torch = importlib.import_module("torch")
        self._transformers = None  # loaded on first use

        default_cpu_model = os.getenv("ASR_MODEL_ID_CPU", "openai/whisper-small")
        default_gpu_model = os.getenv(
            "ASR_MODEL_ID_GPU", "openai/whisper-large-v3-turbo"
        )
        self._model_id = model_id or (
            default_gpu_model if self._torch.cuda.is_available() else default_cpu_model
        )

        if eager_load:
            _ = self.get_pipeline()

    def get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            # Import heavy deps only when needed
            import importlib

            self._transformers = importlib.import_module("transformers")

            device_str = "cuda:0" if self._torch.cuda.is_available() else "cpu"
            torch_dtype = (
                self._torch.float16 if self._torch.cuda.is_available() else self._torch.float32
            )

            t0 = time.time()
            print(
                f"[ASR] Loading model '{self._model_id}' on {device_str} (dtype={torch_dtype}) …"
            )

            AutoModelForSpeechSeq2Seq = getattr(
                self._transformers, "AutoModelForSpeechSeq2Seq"
            )
            AutoProcessor = getattr(self._transformers, "AutoProcessor")
            hf_pipeline = getattr(self._transformers, "pipeline")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=self._local_only,
            )
            model.to(device_str)
            processor = AutoProcessor.from_pretrained(
                self._model_id, local_files_only=self._local_only
            )
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_str,
            )
            print(f"[ASR] Loaded in {time.time() - t0:.1f}s")
            return self._pipeline

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 audio array at the given sample_rate."""
        try:
            pipe = self.get_pipeline()
            result = pipe({"array": audio.astype(np.float32), "sampling_rate": sample_rate})
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return (text or "").strip()
        except Exception as e:
            print(f"[ASR] transcription failed: {e}")
            return ""

    def warmup(self, audio: Optional[np.ndarray], sample_rate: int) -> None:
        """Run a dummy inference to warm up the pipeline."""
        try:
            pipe = self.get_pipeline()
            if audio is None or audio.size == 0:
                audio = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
            _ = pipe({"array": audio.astype(np.float32), "sampling_rate": sample_rate})
        except Exception as e:
            print(f"[WARMUP] ASR inference failed: {e}")


