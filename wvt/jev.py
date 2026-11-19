"""Jev 1.13 System One as an optional decision-model router for the assistant (``assistant.router: "jev"``).

The rule-based :func:`wvt.assistant.parse_intent` is fast and predictable, but it never misses: an utterance it
does not know becomes ``Intent("chat")``.  When that happens the assistant may ask Jev one question - is this a
command at all (``noul``), and which command is it (``choice``) - and map the answer back to an intent.  Any
failure, timeout or low confidence leaves the local chat/LLM path untouched, so the router can never break a
command it did not invent.

One POST to ``/systemone`` with the plain text of the current utterance (no chat log, no quest data); the key
travels in the ``Authorization: Bearer`` header, never in the URL and never in a log.  Latency is ~350-440 ms;
``jev-1.13`` costs $0.042 per 1M input tokens (output free), ``jev-1.13-free`` is a limited-time free model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .gemini import CloudError, rest_seconds

log = logging.getLogger(__name__)

DEFAULT_URL = "https://opencode.ai/inference/systemone/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"

_IS_COMMAND = ("Does the player ask the assistant to do something in the game (group actions, chat summary, "
               "quest log, route plan, stats, items, reminders) rather than asking a general game-knowledge "
               "question or making small talk?")
_INTENT = ("Pick the one command that best matches the player's Polish utterance. Pick 'other' for ordinary "
           "conversation or a general game-knowledge question.")


class JevError(CloudError):
    """Jev failed (network, limit, bad key, unreadable answer): the local path answers instead."""

    def __init__(self, status: int, reason: str):
        super().__init__(f"Jev {status or 'connection'}: {reason}", status)
        self.status = status


@dataclass
class JevDecision:
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_command: bool
    ms: int
    usage: dict = field(default_factory=dict)
    provider: str = ""


def _endpoint(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/systemone") or url.endswith("/decisions"):
        return url
    if url.endswith("/api") or url.endswith("/api/v1"):
        return url + "/systemone"
    return url + "/systemone"


def _reason(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return r.reason_phrase
    if not isinstance(data, dict):
        return r.reason_phrase
    for key in ("detail", "message", "error"):
        val = data.get(key)
        if isinstance(val, dict):
            val = val.get("message") or val.get("detail") or val.get("type")
        if val:
            return str(val)[:160].strip() or r.reason_phrase
    return r.reason_phrase


def _is_true(value) -> bool:
    """The ``noul`` answer as a bool: a bool, ``{"value": ...}`` / ``{"noul": ...}``, or "true"/"yes"/"1"."""
    if isinstance(value, dict):
        value = value.get("value") or value.get("noul")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class JevClient:
    """One decision-model endpoint: ``classify`` returns a :class:`JevDecision` or None (no confident label)."""

    supports_thinking = False

    def __init__(self, api_key: str, model: str = "jev-1.13", base_url: str = DEFAULT_URL,
                 timeout_s: float = 2.5, min_confidence: float = 0.55,
                 client: Optional[httpx.Client] = None, provider: str = ""):
        self._key = (api_key or "").strip()
        self.model = model
        self.provider = provider or model
        self.base_url = base_url
        self.url = _endpoint(base_url)
        self.timeout_s = float(timeout_s)
        self.min_confidence = float(min_confidence)
        self._client = client or httpx.Client(timeout=self.timeout_s)
        self._resting_until = 0.0
        self.calls = 0
        self.failures = 0
        self.last_ms = 0

    @property
    def model_name(self) -> str:
        return self.model

    def available(self) -> bool:
        return bool(self._key) and time.monotonic() >= self._resting_until

    def _rest(self, status: int, reason: str = "", retry_after=None) -> None:
        pause = rest_seconds(status, reason)
        try:
            hinted = int(float(retry_after)) if retry_after is not None else 0
        except (TypeError, ValueError):
            hinted = 0
        if hinted > 0:
            pause = hinted
        self._resting_until = time.monotonic() + pause
        if pause >= 600:
            log.warning("Jev answered %s - check the key, billing and the model name; the local rules answer "
                        "for the next %d minutes", status, pause // 60)

    def classify(self, text: str, criteria: dict[str, str], instructions: Optional[str] = None,
                 min_confidence: Optional[float] = None) -> Optional[JevDecision]:
        text = (text or "").strip()
        if not text or not criteria:
            return None
        if not self.available():
            return None
        body = {
            "model": self.model,
            "state": text,
            "questions": {
                "is_command": {"type": "noul", "instructions": _IS_COMMAND},
                "intent": {"type": "choice", "instructions": instructions or _INTENT,
                           "criteria": dict(criteria)},
            },
        }
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        self.calls += 1
        t0 = time.perf_counter()
        try:
            r = self._client.post(self.url, json=body, timeout=self.timeout_s, headers=headers)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            self.failures += 1
            self._rest(0)
            raise JevError(0, type(exc).__name__) from None
        self.last_ms = round((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            self.failures += 1
            reason = _reason(r)
            self._rest(r.status_code, reason, r.headers.get("Retry-After"))
            raise JevError(r.status_code, reason)
        try:
            data = r.json()
            answers = data["answers"]
            usage = data.get("usage", {}) or {}
            if not isinstance(answers, dict):
                raise TypeError("answers is not an object")
        except (ValueError, TypeError, AttributeError, KeyError):
            self.failures += 1
            raise JevError(0, "unreadable answer") from None
        is_command = _is_true((answers.get("is_command") or {}))
        if not is_command:
            return JevDecision("chat", 1.0, {}, False, self.last_ms, usage, self.provider)
        intent = answers.get("intent") or {}
        if not isinstance(intent, dict):
            intent = {}
        label = str(intent.get("choice") or intent.get("value") or "").strip().lower()
        try:
            conf = float(intent.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        probs: dict[str, float] = {}
        raw = intent.get("probabilities") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    probs[str(k).lower()] = float(v)
                except (TypeError, ValueError):
                    continue
        threshold = self.min_confidence if min_confidence is None else float(min_confidence)
        if not label or label == "other" or conf < threshold:
            return None
        return JevDecision(label, conf, probs, True, self.last_ms, usage, self.provider)

    def close(self) -> None:
        self._client.close()


class JevChain:
    """Jev providers tried in order: the first confident decision wins, failures fall through."""

    def __init__(self, clients: list[JevClient]):
        self.clients = [c for c in clients if c is not None]

    @property
    def model_name(self) -> str:
        return " -> ".join(f"{c.model_name} ({c.provider})" for c in self.clients)

    def available(self) -> bool:
        return any(c.available() for c in self.clients)

    def classify(self, text: str, criteria: dict[str, str], instructions: Optional[str] = None,
                 min_confidence: Optional[float] = None) -> Optional[JevDecision]:
        for c in self.clients:
            if not c.available():
                continue
            try:
                decision = c.classify(text, criteria, instructions, min_confidence)
            except Exception as exc:
                log.info("Jev %s failed (%s) - trying the next provider", c.provider, exc)
                continue
            if decision is not None:
                return decision
        return None

    def close(self) -> None:
        for c in self.clients:
            try:
                c.close()
            except Exception:
                log.info("closing Jev %s failed", c.provider)


def jev_router(cfg):
    """The Jev chain when ``assistant.router`` is "jev" and at least one key is in .env; otherwise None."""
    if str(cfg.get("assistant.router", "local") or "local").lower() != "jev":
        return None
    j = cfg.get("assistant.jev", {}) or {}
    legacy = {
        "base_url": j.get("base_url", DEFAULT_URL),
        "api_key_env": j.get("api_key_env", "OPENCODE_API_KEY"),
        "model": j.get("model", "jev-1.13"),
    }
    defaults = {
        "openrouter": {"base_url": OPENROUTER_URL, "api_key_env": "OPENROUTER_API_KEY",
                       "model": "typesafe/jev-1.14"},
        "opencode": legacy,
    }
    order = j.get("order") or ["openrouter", "opencode"]
    if isinstance(order, str):
        order = order.split(",")
    timeout_s = float(j.get("timeout_s", 2.5))
    min_confidence = float(j.get("min_confidence", 0.55))
    from .envfile import secret

    clients: list[JevClient] = []
    for raw_name in order:
        name = str(raw_name or "").strip().lower()
        base = defaults.get(name)
        if base is None:
            log.warning("assistant.jev.order: unknown provider %r - skipped", name)
            continue
        block = j.get(name, {}) or {}
        base_url = block.get("base_url", base["base_url"])
        env_name = block.get("api_key_env", base["api_key_env"])
        model = block.get("model", base["model"])
        block_timeout = float(block.get("timeout_s", timeout_s))
        block_confidence = float(block.get("min_confidence", min_confidence))
        key = secret(env_name)
        if not key:
            log.info("assistant.jev: there is no %s in .env - %s skipped", env_name, name)
            continue
        try:
            clients.append(JevClient(key, model, base_url, block_timeout, block_confidence, provider=name))
        except Exception as exc:  # a bad config must not stop startup
            log.warning("the Jev provider %s could not be set up (%s) - skipped", name, exc)
    clients = [c for c in clients if c is not None]
    if not clients:
        log.warning("assistant.router is jev, but no Jev provider has a key in .env - the local rules answer")
        return None
    return JevChain(clients)
