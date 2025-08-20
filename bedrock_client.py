import os
import importlib
from typing import Optional, List, Dict


class BedrockAssistant:
    """Thin wrapper around AWS Bedrock Messages API (Anthropic models).

    - Lazily creates the bedrock-runtime client on first use
    - Parses common Anthropic responses on Bedrock
    - Exposes two conveniences: check_wake() and chat_completion()
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.debug: bool = os.getenv("BEDROCK_DEBUG", "").lower() in ("1", "true", "yes", "on")
        self.model_id: str = model_id or os.getenv("BEDROCK_MODEL_ID", "apac.anthropic.claude-sonnet-4-20250514-v1:0")
        self.region: str = region or os.getenv("BEDROCK_REGION", "ap-southeast-2")
        self.max_tokens: int = int(max_tokens or os.getenv("BEDROCK_MAX_TOKENS", "256"))
        self.temperature: float = float(temperature or os.getenv("BEDROCK_TEMPERATURE", "0.7"))
        self.anthropic_version: str = os.getenv("BEDROCK_ANTHROPIC_VERSION", "bedrock-2023-05-31")

        env_enabled = os.getenv("BEDROCK_ENABLED", "").lower() in ("1", "true", "yes", "on")
        self.enabled: bool = env_enabled if enabled is None else bool(enabled)

        self._boto3 = None
        self._botocore_exceptions = None
        self._client = None

        # In-memory chat history as Bedrock Messages API entries
        # Each entry: {"role": "user"|"assistant", "content": [{"type":"text","text": ...}, ...]}
        self.history: List[Dict] = []
        try:
            self.max_history_messages: int = int(os.getenv("BEDROCK_MAX_HISTORY", "20"))
        except ValueError:
            self.max_history_messages = 20

        if self.debug:
            aws_prof = os.getenv("AWS_PROFILE")
            aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            print(
                f"[BEDROCK DEBUG] init: enabled={self.enabled} model_id={self.model_id} "
                f"region={self.region} max_tokens={self.max_tokens} temperature={self.temperature} "
                f"AWS_PROFILE={aws_prof} AWS_REGION={aws_region}"
            )

        if self.enabled:
            boto3_spec = importlib.util.find_spec("boto3")
            botocore_spec = importlib.util.find_spec("botocore.exceptions")
            if boto3_spec is None or botocore_spec is None:
                print("[BEDROCK] Disabled (boto3/botocore not installed)")
                self.enabled = False
            else:
                self._boto3 = importlib.import_module("boto3")
                self._botocore_exceptions = importlib.import_module("botocore.exceptions")
                if self.debug:
                    print("[BEDROCK DEBUG] boto3/botocore modules loaded")
                    self._debug_sanity()

    def _debug_sanity(self) -> None:
        """Best-effort control-plane checks to aid debugging (model/region)."""
        try:
            bedrock_ctl = self._boto3.client("bedrock", region_name=self.region)
        except (AttributeError, self._botocore_exceptions.BotoCoreError, self._botocore_exceptions.ClientError) as e:
            print(f"[BEDROCK DEBUG] control-plane client init failed: {type(e).__name__}: {e}")
            return

        try:
            resp = bedrock_ctl.list_foundation_models(byProvider="Anthropic")
            summaries = resp.get("modelSummaries", [])
            available = [m.get("modelId") for m in summaries if isinstance(m, dict)]
            print(f"[BEDROCK DEBUG] available Anthropic models in {self.region}: {available[:10]}")
            found = self.model_id in set(available)
            print(f"[BEDROCK DEBUG] model_id found={found} model_id={self.model_id}")
            if not found:
                # Surface near matches
                suggestions = [mid for mid in available if isinstance(mid, str) and mid.startswith("anthropic.")]
                print(f"[BEDROCK DEBUG] suggestions (starts with 'anthropic.'): {suggestions[:10]}")
        except (self._botocore_exceptions.BotoCoreError, self._botocore_exceptions.ClientError, KeyError, TypeError, AttributeError) as e:
            print(f"[BEDROCK DEBUG] list_foundation_models failed: {type(e).__name__}: {e}")

    def _ensure_client(self) -> bool:
        if not self.enabled or self._boto3 is None or self._botocore_exceptions is None:
            return False
        if self._client is None:
            try:
                self._client = self._boto3.client("bedrock-runtime", region_name=self.region)
                if self.debug:
                    print(f"[BEDROCK DEBUG] bedrock-runtime client created in region {self.region}")
            except (
                self._botocore_exceptions.BotoCoreError,
                self._botocore_exceptions.ClientError,
                self._botocore_exceptions.NoCredentialsError,
            ):
                print("[BEDROCK] Disabled (could not create client; check AWS credentials/region)")
                self.enabled = False
                return False
        return True

    def _invoke_messages(self, messages: List[Dict], system_text: Optional[str] = None) -> Optional[str]:
        if not self._ensure_client():
            return None
        try:
            import json
            body: Dict = {
                "anthropic_version": self.anthropic_version,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages,
            }
            if system_text:
                body["system"] = [{"type": "text", "text": system_text}]

            if self.debug:
                # Log a compact preview of the request
                preview = None
                try:
                    preview = json.dumps(body)[:600]
                except (TypeError, ValueError):
                    preview = "<unserializable-request-body>"
                print(
                    f"[BEDROCK DEBUG] invoke: model_id={self.model_id} region={self.region} "
                    f"accept=application/json contentType=application/json body_preview={preview}"
                )

            response = self._client.invoke_model(
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
                body=json.dumps(body),
            )
        except (
            self._botocore_exceptions.BotoCoreError,
            self._botocore_exceptions.ClientError,
        ) as e:
            # Detailed error diagnostics
            print(f"[BEDROCK] request failed: {type(e).__name__}: {e}")
            # Best-effort body serialization for debugging
            try:
                print(f"[BEDROCK DEBUG] last_request_body={json.dumps(body)[:1000]}")
            except (TypeError, ValueError):
                print("[BEDROCK DEBUG] last_request_body=<unserializable>")
            if hasattr(e, "response") and isinstance(e.response, dict):
                meta = e.response.get("ResponseMetadata")
                err = e.response.get("Error")
                print(f"[BEDROCK DEBUG] response_metadata={meta}")
                if isinstance(err, dict):
                    code = err.get("Code")
                    msg = err.get("Message")
                    print(f"[BEDROCK DEBUG] error_code={code} error_message={msg}")
            return None

        try:
            response_body = response.get("body")
            if response_body is None:
                return None
            import json
            raw_text = response_body.read().decode("utf-8")
            if self.debug:
                print(f"[BEDROCK DEBUG] raw_response_text_preview={raw_text[:1000]}")
            response_json = json.loads(raw_text)
            output_text: Optional[str] = None
            if isinstance(response_json, dict):
                # Preferred shape: top-level content list
                content = response_json.get("content")
                if isinstance(content, list):
                    output_text = "".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if not output_text and "output" in response_json and isinstance(response_json["output"], dict):
                    # Fallback older shape
                    message = response_json["output"].get("message", {})
                    content2 = message.get("content", []) if isinstance(message, dict) else []
                    output_text = "".join(
                        part.get("text", "") for part in content2
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
            if self.debug:
                print(f"[BEDROCK DEBUG] parsed_output_text_preview={(output_text or '')[:300]}")
            return output_text or None
        except (ValueError, AttributeError, KeyError, TypeError) as e:
            print(f"[BEDROCK] response parse error: {type(e).__name__}: {e}")
            return None

    def check_wake(self, user_text: str) -> bool:
        """Heuristically determine if user_text is effectively the wake phrase."""
        if not self.enabled:
            return False
        instruction = (
            "You classify if the user text indicates the wake phrase 'hello computer'. "
            "Treat minor misrecognitions, filler words around it, or swapped order as WAKE only if clearly intended. "
            "Respond with exactly one token: WAKE or NO."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Text: {user_text}"}],
            }
        ]
        out = self._invoke_messages(messages, system_text=instruction)
        if not out:
            return False
        normalized = out.strip().lower()
        if self.debug:
            print(f"[BEDROCK DEBUG] wake_check_normalized={normalized}")
        return normalized.startswith("wake") or normalized == "yes"

    def chat_completion(self, user_texts: List[str], system_text: Optional[str] = None) -> Optional[str]:
        """
        Send one chat turn consisting of an array of user text strings.

        - Maintains conversation state by appending the user turn and assistant response
          to internal history which is sent on subsequent calls.
        - Pass system_text to set instruction for this call (not persisted).
        """
        if not self.enabled:
            return None

        # Build a single user message with multiple text parts
        user_message: Dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": t}
                for t in user_texts
                if isinstance(t, str) and t.strip()
            ],
        }
        if not user_message["content"]:
            return None

        # Bound history size to avoid unbounded context growth
        def _bounded_history(hist: List[Dict]) -> List[Dict]:
            if self.max_history_messages <= 0:
                return hist
            return hist[-self.max_history_messages :]

        messages_to_send: List[Dict] = _bounded_history(self.history) + [user_message]
        assistant_text = self._invoke_messages(messages_to_send, system_text=system_text)
        if assistant_text is None:
            return None

        # Persist the successful turn in history
        self.history = _bounded_history(self.history + [user_message, {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        }])
        return assistant_text

    def reset_conversation(self) -> None:
        """Clear in-memory chat history for a fresh conversation."""
        self.history = []


