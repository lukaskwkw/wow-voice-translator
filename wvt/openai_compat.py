"""Any OpenAI-compatible cloud chat API for the assistant only (``assistant.llm: "openai"``).

One POST to ``base_url`` + ``/chat/completions`` with the messages as they are (OpenAI format), no SDK.  The
key travels in the ``Authorization: Bearer`` header, never in the URL and never in a log.  Any failure costs only
that one answer: :class:`wvt.gemini.FallbackLLM` asks the local model instead, and after a rate limit or a key
error the cloud rests for a while (the same ``rest_seconds`` table as Gemini).

Translations, voices, speech recognition and the knowledge base stay local.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from .gemini import CloudError, rest_seconds

log = logging.getLogger(__name__)


class OpenAICompatError(CloudError):
    def __init__(self, status: int, reason: str):
        super().__init__(f"OpenAI-compatible {status or 'connection'}: {reason}", status)
        self.status = status


def _endpoint(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def _reason(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return r.reason_phrase
    err = data.get("error") if isinstance(data, dict) else None
    msg = ""
    if isinstance(err, dict):
        msg = str(err.get("message", ""))
    elif isinstance(err, str):
        msg = err
    if not msg and isinstance(data, dict):
        msg = str(data.get("message", ""))
    return (msg[:160] or r.reason_phrase).strip()


class OpenAICompatLLM:
    """The ``chat`` interface of :class:`wvt.llm.LocalLLM`, answered by an OpenAI-compatible cloud API."""

    supports_thinking = False

    def __init__(self, api_key: str, model: str, base_url: str, timeout_s: float = 20.0,
                 max_output_tokens: int = 4096, client: Optional[httpx.Client] = None,
                 extra_headers: Optional[dict] = None):
        self._key = (api_key or "").strip()
        self.model = model
        self.base_url = base_url
        self.url = _endpoint(base_url)
        self.timeout_s = float(timeout_s)
        self.max_output_tokens = int(max_output_tokens)
        self._client = client or httpx.Client(timeout=self.timeout_s)
        self._extra_headers = dict(extra_headers or {})
        self._resting_until = 0.0
        self.calls = 0
        self.failures = 0
        self.last_ms = 0

    @property
    def model_name(self) -> str:
        return self.model

    def available(self) -> bool:
        return bool(self._key) and time.monotonic() >= self._resting_until

    def _rest(self, status: int, reason: str = "") -> None:
        pause = rest_seconds(status, reason)
        self._resting_until = time.monotonic() + pause
        if pause >= 600:
            log.warning("the cloud model answered %s - check the key, billing and the model name; the local model "
                        "answers for the next %d minutes", status, pause // 60)

    def chat(self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.0,
             stop: list[str] | None = None, timeout: float | None = None,
             thinking: Optional[str] = None) -> tuple[str, dict]:
        """``thinking`` is ignored: an OpenAI-compatible endpoint has no thinking level."""
        body: dict = {"model": self.model, "messages": messages, "stream": False,
                      "max_tokens": int(max_tokens or self.max_output_tokens)}
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop"] = stop
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        headers.update(self._extra_headers)
        self.calls += 1
        t0 = time.perf_counter()
        try:
            r = self._client.post(self.url, json=body, timeout=timeout or self.timeout_s, headers=headers)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            self.failures += 1
            self._rest(0)
            raise OpenAICompatError(0, type(exc).__name__) from None
        self.last_ms = round((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            self.failures += 1
            reason = _reason(r)
            self._rest(r.status_code, reason)
            raise OpenAICompatError(r.status_code, reason)
        try:
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            if not isinstance(choice, dict):
                raise TypeError("a choice is not an object")
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(str(p.get("text", "")) for p in content
                               if isinstance(p, dict) and p.get("type", "text") == "text").strip()
            else:
                text = str(content or "").strip()
            finish = choice.get("finish_reason", "")
        except (ValueError, TypeError, AttributeError):
            self.failures += 1
            raise OpenAICompatError(0, "unreadable answer") from None
        if not text:
            self.failures += 1
            raise OpenAICompatError(200, f"empty answer ({finish or 'no choice'})")
        return text, {"ms": self.last_ms, "finish": finish, "usage": data.get("usage", {}),
                      "model": data.get("model", self.model)}

    def close(self) -> None:
        self._client.close()
