import numpy as np
import torch
from dataclasses import dataclass

from silero_vad import load_silero_vad, VADIterator


@dataclass
class VoiceActivationConfig:
    sample_rate: int = 16000
    threshold: float = 0.5
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 100
    frame_ms: int = 32  # audio callback frame size in milliseconds


class SileroVoiceActivation:
    """
    Lightweight voice activation using Silero VAD's streaming iterator.

    Emits simple events based on the internal `triggered` state of VADIterator:
    - 'speech_start' when entering speech
    - 'speech_end' when leaving speech

    The implementation mirrors the activation behavior used in pipecat's
    Silero VAD analyzer, where the iterator's state machine handles hysteresis
    (threshold/min_silence) and we surface start/stop events.
    """

    def __init__(self, config: VoiceActivationConfig | None = None):
        self.config = config or VoiceActivationConfig()

        # Load Silero VAD model (ONNX for performance/portability)
        self.model = load_silero_vad(onnx=True)

        # Create the streaming iterator with gating parameters
        self.vad_iterator = VADIterator(
            self.model,
            threshold=self.config.threshold,  # Confidence threshold for speech detection (0.0-1.0)
            sampling_rate=self.config.sample_rate,  # Audio sample rate in Hz
            min_silence_duration_ms=self.config.min_silence_duration_ms,  # Minimum silence duration to trigger speech_end
            speech_pad_ms=self.config.speech_pad_ms,  # Padding added before/after detected speech segments
        )

        self._prev_triggered = False

    def reset(self) -> None:
        try:
            # Some backends expose reset_states; ignore if not available
            self.model.reset_states()
        except Exception:
            pass
        self.vad_iterator.reset_states()
        self._prev_triggered = False

    @property
    def is_active(self) -> bool:
        return getattr(self.vad_iterator, "triggered", False)

    def process_frame(self, frame: np.ndarray) -> str | None:
        """
        Feed a mono float32 frame (shape [N]) at `sample_rate` into the detector.

        Returns one of:
          - 'speech_start'
          - 'speech_end'
          - None (no boundary event this frame)
        """
        if frame.ndim > 1:
            # Downmix to mono if needed
            frame = np.mean(frame, axis=1)

        # Ensure contiguous float32 tensor for the VAD iterator
        audio_tensor = torch.from_numpy(np.ascontiguousarray(frame)).float()

        # Advance the iterator's internal state machine on this frame
        _ = self.vad_iterator(audio_tensor)

        is_triggered = getattr(self.vad_iterator, "triggered", False)

        event: str | None = None
        if not self._prev_triggered and is_triggered:
            event = "speech_start"
        elif self._prev_triggered and not is_triggered:
            event = "speech_end"

        self._prev_triggered = is_triggered
        return event


