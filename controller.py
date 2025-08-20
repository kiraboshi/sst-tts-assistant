from typing import Optional, List, Callable

import numpy as np

from regex_utils import compile_phrase_regex


class ConversationController:
    """
    Manages wake/listen/reset/stop behavior and invokes ASR + Assistant.
    """

    MODE_IDLE = "idle"
    MODE_LISTEN = "listen"

    def __init__(
        self,
        start_phrase: str,
        stop_phrase: str,
        sample_rate: int,
        asr_transcribe: Callable[[np.ndarray, int], str],
        assistant,
        on_beep_on: Optional[Callable[[], None]] = None,
        on_beep_off: Optional[Callable[[], None]] = None,
        on_beep_reset: Optional[Callable[[], None]] = None,
        llm_checks_enabled: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self._asr_transcribe = asr_transcribe
        self._assistant = assistant
        self._on_beep_on = on_beep_on
        self._on_beep_off = on_beep_off
        self._on_beep_reset = on_beep_reset
        self._llm_checks_enabled = llm_checks_enabled

        self.mode = self.MODE_IDLE
        self.transcript_parts: List[str] = []

        self._wake_re = compile_phrase_regex(start_phrase)
        self._stop_re = compile_phrase_regex(stop_phrase)
        self._reset_re = compile_phrase_regex("reset")
        self._stop_phrase = stop_phrase

    def handle_segment(self, segment: np.ndarray) -> None:
        text = self._asr_transcribe(segment, self.sample_rate)
        if not text:
            return

        # Idle: look for wake command
        if self.mode == self.MODE_IDLE:
            print(f"Idle: {text}")
            m = self._wake_re.search(text)
            should_wake = False
            if m:
                should_wake = True
                trailing = text[m.end():].strip()
            else:
                trailing = ""
                if getattr(self._assistant, "enabled", False) and self._llm_checks_enabled:
                    should_wake = self._assistant.check_wake(text)

            if should_wake:
                self.mode = self.MODE_LISTEN
                self.transcript_parts.clear()
                if getattr(self._assistant, "enabled", False):
                    self._assistant.reset_conversation()
                if trailing:
                    self.transcript_parts.append(trailing)
                print(
                    f">>> Listen mode ON. Say '{self._stop_phrase}' to finish."
                )
                if self._on_beep_on:
                    self._on_beep_on()
            return

        # Listen mode
        m_reset = self._reset_re.search(text)
        if m_reset:
            self.transcript_parts.clear()
            if getattr(self._assistant, "enabled", False):
                self._assistant.reset_conversation()
            print(
                ">>> Conversation reset."
            )
            if self._on_beep_reset:
                self._on_beep_reset()
            return

        m_stop = self._stop_re.search(text)
        should_sleep = False
        leading: Optional[str] = None
        if m_stop:
            should_sleep = True
            leading = text[: m_stop.start()].strip()
        else:
            if getattr(self._assistant, "enabled", False) and self._llm_checks_enabled:
                should_sleep = self._assistant.check_sleep(text, self._stop_phrase)

        if should_sleep:
            if leading:
                self.transcript_parts.append(leading)
            final_text = " ".join(
                p for p in (part.strip() for part in self.transcript_parts) if p
            )
            print(final_text if final_text else "")
            self.transcript_parts.clear()
            self.mode = self.MODE_IDLE
            print(
                ">>> Listen mode OFF."
            )
            if self._on_beep_off:
                self._on_beep_off()
            return

        print(f"Text: {text}")
        if getattr(self._assistant, "enabled", False):
            response = self._assistant.chat_completion(
                [text],
                system_text="You are a helpful assistant. Answer the user query in one sentence.",
            )
            if response:
                print(f"[OPENAI] {response}")
                speak = getattr(self._assistant, "speak", None)
                if callable(speak):
                    try:
                        speak(response)
                    except Exception:
                        pass

        self.transcript_parts.append(text)


