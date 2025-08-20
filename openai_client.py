import os
import importlib
import socket
from urllib.parse import urlparse
from typing import Optional, List, Dict


class OpenAIAssistant:
    """Thin wrapper around OpenAI's Chat Completions API.

    - Lazily creates the OpenAI client on first use
    - Maintains simple chat history
    - Exposes two conveniences: check_wake() and chat_completion()
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.debug: bool = os.getenv("OPENAI_DEBUG", "").lower() in ("1", "true", "yes", "on")

        # Model and runtime settings
        self.model_id: str = model_id or os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        self.max_tokens: int = int(max_tokens or os.getenv("OPENAI_MAX_TOKENS", "256"))
        self.temperature: float = float(temperature or os.getenv("OPENAI_TEMPERATURE", "0.7"))
        # Client options (official library): timeout (float or httpx.Timeout), retries per-request via with_options
        timeout_env = os.getenv("OPENAI_TIMEOUT")
        self.timeout: Optional[float] = None
        try:
            if timeout_env:
                self.timeout = float(timeout_env)
        except ValueError:
            self.timeout = None
        max_retries_env = os.getenv("OPENAI_MAX_RETRIES")
        try:
            self.max_retries: Optional[int] = int(max_retries_env) if max_retries_env else None
        except ValueError:
            self.max_retries = None

        # Helpers to sanitize environment values
        def _clean_env(name: str) -> Optional[str]:
            raw = os.getenv(name)
            if raw is None:
                return None
            cleaned = raw.strip()
            if cleaned != raw and self.debug:
                print(f"[OPENAI DEBUG] sanitized env {name}: trimmed whitespace")
            # Remove any lingering control characters in the value
            if any(ch in cleaned for ch in ("\r", "\n", "\t")):
                cleaned2 = cleaned.replace("\r", "").replace("\n", "").replace("\t", "")
                if cleaned2 != cleaned and self.debug:
                    print(f"[OPENAI DEBUG] sanitized env {name}: removed control characters")
                cleaned = cleaned2
            return cleaned or None

        # API keys and base URL selection
        self.api_key: Optional[str] = _clean_env("OPENAI_API_KEY")
        self.base_url: Optional[str] = base_url or _clean_env("OPENAI_BASE_URL")
        self.provider: str = "openai"
        or_key = _clean_env("OPENROUTER_API_KEY")
        if or_key:
            # Use OpenRouter if its key is provided
            self.api_key = or_key
            self.base_url = "https://openrouter.ai/api/v1"
            if self.debug and (base_url or _clean_env("OPENAI_BASE_URL")):
                print("[OPENAI DEBUG] Overriding provided base_url to OpenRouter endpoint due to OPENROUTER_API_KEY.")
            self.provider = "openrouter"

        # Optional org/project
        self.organization: Optional[str] = _clean_env("OPENAI_ORG_ID")
        self.project: Optional[str] = _clean_env("OPENAI_PROJECT")

        # Enable when explicitly set or when an API key is present
        env_enabled = os.getenv("OPENAI_ENABLED", "").lower() in ("1", "true", "yes", "on")
        self.enabled: bool = env_enabled if enabled is None else bool(enabled)
        if enabled is None and not env_enabled:
            self.enabled = bool(self.api_key)

        self._openai_module = None
        self._client = None

        # In-memory chat history (OpenAI role/content schema, stored as Bedrock-like for parity)
        # Each entry: {"role": "user"|"assistant", "content": [{"type":"text","text": ...}, ...]}
        self.history: List[Dict] = []
        try:
            self.max_history_messages: int = int(os.getenv("OPENAI_MAX_HISTORY", "20"))
        except ValueError:
            self.max_history_messages = 20

        if self.debug:
            print(
                f"[OPENAI DEBUG] init: enabled={self.enabled} provider={self.provider} model_id={self.model_id} "
                f"max_tokens={self.max_tokens} temperature={self.temperature} base_url={self.base_url} "
                f"api_key_present={'yes' if self.api_key else 'no'}"
            )

        if self.enabled:
            openai_spec = importlib.util.find_spec("openai")
            if openai_spec is None:
                print("[OPENAI] Disabled (openai package not installed)")
                self.enabled = False
            else:
                self._openai_module = importlib.import_module("openai")
                if self.debug:
                    print("[OPENAI DEBUG] openai module loaded")

    def _effective_base_url(self) -> str:
        # Prefer explicit base URL, otherwise default to the official endpoint
        return self.base_url or "https://api.openai.com/v1"

    def _redact(self, text: str) -> str:
        try:
            import re
            redacted = text
            # Redact common bearer token appearances
            redacted = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer <redacted>", redacted)
            # Redact OpenAI/Alt keys like sk-..., sk-or-v1-...
            redacted = re.sub(r"sk-[A-Za-z0-9\-]{8,}", "sk-<redacted>", redacted)
            redacted = re.sub(r"sk-or-v1-[A-Za-z0-9]+", "sk-or-v1-<redacted>", redacted)
            # Redact potential session or personal tokens
            redacted = re.sub(r"sess_[A-Za-z0-9\-_.]+", "sess_<redacted>", redacted)
            redacted = re.sub(r"pat_[A-Za-z0-9\-_.]+", "pat_<redacted>", redacted)
            return redacted
        except Exception:
            return text

    def _log_network_context(self, context_note: str = "") -> None:
        """Emit detailed diagnostics helpful for debugging connectivity/response issues."""
        try:
            base_url = self._effective_base_url()
            parsed = urlparse(base_url)
            host = parsed.hostname or "<unknown>"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            # Resolve DNS to surface any resolution issues
            resolved_entries: list[str] = []
            try:
                infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
                for fam, _stype, _proto, _canon, sockaddr in infos:
                    ip = sockaddr[0]
                    resolved_entries.append(f"{ip}")
            except Exception as e:
                resolved_entries.append(f"<dns-error: {type(e).__name__}: {e}>")

            raw_proxies = {
                k: os.getenv(k)
                for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY")
                if os.getenv(k)
            }
            # Sanitize proxy values to avoid leaking credentials
            sanitized_proxies: Dict[str, str] = {}
            for k, v in raw_proxies.items():
                if not v:
                    continue
                try:
                    p = urlparse(v)
                    host = p.hostname or "<unknown>"
                    port = p.port or (443 if p.scheme == "https" else 80)
                    sanitized_proxies[k] = f"{p.scheme}://{host}:{port}"
                except Exception:
                    sanitized_proxies[k] = "<set>"

            openai_version = None
            try:
                openai_version = getattr(self._openai_module, "__version__", None)
            except Exception:
                pass

            httpx_version = None
            try:
                import httpx  # type: ignore
                httpx_version = getattr(httpx, "__version__", None)
            except Exception:
                pass

            print(
                "[OPENAI DEBUG] network context" + (f" ({context_note})" if context_note else "")
            )
            print(
                f"[OPENAI DEBUG] base_url={base_url} provider={self.provider} model_id={self.model_id} host={host} port={port} resolved_ips={resolved_entries}"
            )
            print(
                f"[OPENAI DEBUG] sdk_version=openai:{openai_version or '<unknown>'} httpx:{httpx_version or '<unknown>'}"
            )
            if sanitized_proxies:
                print(f"[OPENAI DEBUG] proxies={sanitized_proxies}")
            else:
                print("[OPENAI DEBUG] proxies=<none>")
            # SSL / platform context
            ssl_env = {k: os.getenv(k) for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE") if os.getenv(k)}
            if ssl_env:
                print(f"[OPENAI DEBUG] ssl_env={{{', '.join(f'{k}=<set>' for k in ssl_env.keys())}}}")
            try:
                import platform, sys as _sys
                print(f"[OPENAI DEBUG] platform={platform.platform()} python={_sys.version.splitlines()[0]}")
            except Exception:
                pass
        except Exception as e:
            print(f"[OPENAI DEBUG] failed emitting network context: {type(e).__name__}: {e}")

    def _ensure_client(self) -> bool:
        if not self.enabled or self._openai_module is None:
            return False
        if not self.api_key:
            print("[OPENAI] Disabled (missing OPENAI_API_KEY or OPENROUTER_API_KEY)")
            self.enabled = False
            return False
        if self._client is None:
            try:
                # New-style OpenAI client (v1+)
                OpenAI = getattr(self._openai_module, "OpenAI")
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                if self.organization:
                    client_kwargs["organization"] = self.organization
                if self.project:
                    client_kwargs["project"] = self.project
                if self.timeout is not None:
                    client_kwargs["timeout"] = self.timeout
                self._client = OpenAI(**client_kwargs)
                if self.debug:
                    print(
                        f"[OPENAI DEBUG] client created timeout={self.timeout}"
                    )
            except Exception as e:
                print(f"[OPENAI] Disabled (client init failed: {type(e).__name__}: {e})")
                self.enabled = False
                return False
        return True

    def _to_openai_messages(self, messages: List[Dict], system_text: Optional[str]) -> List[Dict]:
        openai_messages: List[Dict] = []
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})
        for m in messages:
            role = m.get("role")
            content_parts = m.get("content", [])
            if isinstance(content_parts, list):
                # Concatenate text parts; ignore non-text types for this simple wrapper
                text_segments = [
                    part.get("text", "") for part in content_parts
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content_text = "\n\n".join([t for t in text_segments if t])
            else:
                # Fallback if content is a plain string
                content_text = str(content_parts)
            if role in ("user", "assistant", "system") and content_text:
                openai_messages.append({"role": role, "content": content_text})
        return openai_messages

    def _invoke_messages(self, messages: List[Dict], system_text: Optional[str] = None) -> Optional[str]:
        if not self._ensure_client():
            return None
        try:
            openai_messages = self._to_openai_messages(messages, system_text)
            if self.debug:
                try:
                    import json
                    preview = json.dumps(openai_messages)[:600]
                except Exception:
                    preview = "<unserializable-openai-messages>"
                print(
                    f"[OPENAI DEBUG] invoke: model_id={self.model_id} "
                    f"max_tokens={self.max_tokens} temperature={self.temperature} messages={len(openai_messages)} "
                    f"messages_preview={preview}"
                )
                # Emit network context prior to request as well
                self._log_network_context(context_note="pre-request")

            # Apply per-request options like retries if configured
            request_client = self._client
            if self.max_retries is not None:
                try:
                    request_client = request_client.with_options(max_retries=self.max_retries)
                except Exception:
                    pass

            # OpenRouter recommended headers, if provided
            try:
                if self.provider == "openrouter":
                    ref = os.getenv("OPENROUTER_REFERRER") or os.getenv("OPENROUTER_HTTP_REFERER")
                    ttl = os.getenv("OPENROUTER_TITLE")
                    extra_headers: Dict[str, str] = {}
                    if ref:
                        extra_headers["HTTP-Referer"] = ref.strip()
                    if ttl:
                        extra_headers["X-Title"] = ttl.strip()
                    if extra_headers:
                        request_client = request_client.with_options(extra_headers=extra_headers)
                        if self.debug:
                            print(f"[OPENAI DEBUG] using OpenRouter headers keys={list(extra_headers.keys())}")
            except Exception:
                pass

            # Use raw response in debug mode to expose headers
            use_raw = self.debug or os.getenv("OPENAI_DEBUG_RAW") in ("1", "true", "yes", "on")
            if use_raw:
                try:
                    raw_resp = request_client.chat.completions.with_raw_response.create(
                        model=self.model_id,
                        messages=openai_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    try:
                        headers = getattr(raw_resp, "headers", None)
                        if headers:
                            # Show a few interesting headers
                            rid = headers.get("x-request-id") or headers.get("X-Request-Id")
                            ratelimit = {k: headers.get(k) for k in ("x-ratelimit-remaining", "x-ratelimit-limit", "x-ratelimit-reset") if headers.get(k)}
                            print(f"[OPENAI DEBUG] response headers x-request-id={rid} ratelimit={ratelimit}")
                    except Exception:
                        pass
                    response = raw_resp.parse()
                except Exception:
                    # Fallback to normal call if raw path fails
                    response = request_client.chat.completions.create(
                        model=self.model_id,
                        messages=openai_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
            else:
                response = request_client.chat.completions.create(
                    model=self.model_id,
                    messages=openai_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            # Extract assistant text
            try:
                choice0 = response.choices[0]
                message = getattr(choice0, "message", None)
                if message and getattr(message, "content", None):
                    if self.debug:
                        try:
                            finish_reason = getattr(choice0, "finish_reason", None)
                            usage = getattr(response, "usage", None)
                            rid = getattr(response, "id", None)
                            model = getattr(response, "model", None)
                            print(
                                f"[OPENAI DEBUG] response id={rid} model={model} finish_reason={finish_reason} usage={usage}"
                            )
                            content_preview = str(message.content)[:300]
                            print(f"[OPENAI DEBUG] content_preview={content_preview}")
                        except Exception:
                            try:
                                # Best-effort generic preview
                                print(f"[OPENAI DEBUG] response_preview={str(response)[:500]}")
                            except Exception:
                                pass
                    return message.content
            except Exception:
                pass

            # Fallbacks if SDK returns dict-like
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
            return None
        except Exception as e:
            # General error logging with extra diagnostics
            try:
                safe_err = self._redact(str(e))
            except Exception:
                safe_err = str(e)
            print(f"[OPENAI] request failed: {type(e).__name__}: {safe_err}")
            try:
                # Exception attributes common in OpenAI SDK / httpx stack
                status_code = getattr(e, "status_code", None)
                request_id = getattr(e, "request_id", None)
                code = getattr(e, "code", None)
                print(
                    f"[OPENAI DEBUG] error_attrs status_code={status_code} code={code} request_id={request_id}"
                )

                # Response body/headers, if present
                resp = getattr(e, "response", None)
                if resp is not None:
                    # Try httpx-like interface first
                    headers = None
                    text_preview = None
                    try:
                        headers = getattr(resp, "headers", None)
                        if headers and hasattr(headers, "get"):
                            rid = headers.get("x-request-id") or headers.get("X-Request-Id")
                            if rid and not request_id:
                                print(f"[OPENAI DEBUG] x-request-id={rid}")
                        if hasattr(resp, "text"):
                            raw_text = resp.text  # type: ignore[attr-defined]
                            text_preview = (raw_text or "")[:1000]
                        elif hasattr(resp, "json"):
                            try:
                                import json
                                text_preview = json.dumps(resp.json())[:1000]
                            except Exception:
                                text_preview = "<unreadable-json>"
                        else:
                            text_preview = str(resp)[:1000]
                    except Exception as sub_e:
                        text_preview = f"<response-inspect-error: {type(sub_e).__name__}: {sub_e}>"

                    print(f"[OPENAI DEBUG] response_preview={text_preview}")

                # Underlying cause (connectivity/DNS/TLS)
                cause = getattr(e, "__cause__", None)
                if cause is not None:
                    try:
                        safe_cause = self._redact(str(cause))
                    except Exception:
                        safe_cause = str(cause)
                    print(f"[OPENAI DEBUG] cause={type(cause).__name__}: {safe_cause}")
                    deeper = getattr(cause, "__cause__", None)
                    if deeper is not None:
                        try:
                            safe_deeper = self._redact(str(deeper))
                        except Exception:
                            safe_deeper = str(deeper)
                        print(f"[OPENAI DEBUG] cause.cause={type(deeper).__name__}: {safe_deeper}")
            except Exception as log_e:
                print(f"[OPENAI DEBUG] error while logging failure details: {type(log_e).__name__}: {log_e}")

            # Always print network context on failure to aid debugging
            self._log_network_context(context_note="on-exception")
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
            print(f"[OPENAI DEBUG] wake_check_normalized={normalized}")
        return normalized.startswith("wake") or normalized == "yes"

    def check_sleep(self, user_text: str, stop_phrase: str) -> bool:
        """Heuristically determine if user_text indicates the provided stop phrase."""
        if not self.enabled:
            return False
        instruction = (
            "You classify if the user text indicates the stop phrase provided by the application. "
            "Treat minor misrecognitions, filler words around it, or swapped order as SLEEP only if clearly intended. "
            "Respond with exactly one token: SLEEP or NO.\n\n"
            f"Stop phrase: '{stop_phrase}'"
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
            print(f"[OPENAI DEBUG] sleep_check_normalized={normalized}")
        return normalized.startswith("sleep") or normalized == "yes"

    def chat_completion(self, user_texts: List[str], system_text: Optional[str] = None) -> Optional[str]:
        """
        Send one chat turn consisting of an array of user text strings.

        - Maintains conversation state by appending the user turn and assistant response
          to internal history which is sent on subsequent calls.
        - Pass system_text to set instruction for this call (not persisted).
        """
        if not self.enabled:
            return None

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

        def _bounded_history(hist: List[Dict]) -> List[Dict]:
            if self.max_history_messages <= 0:
                return hist
            return hist[-self.max_history_messages :]

        messages_to_send: List[Dict] = _bounded_history(self.history) + [user_message]
        assistant_text = self._invoke_messages(messages_to_send, system_text=system_text)
        if assistant_text is None:
            return None

        # Persist the successful turn in history
        self.history = _bounded_history(self.history + [
            user_message,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ])
        return assistant_text

    def reset_conversation(self) -> None:
        """Clear in-memory chat history for a fresh conversation."""
        self.history = []



