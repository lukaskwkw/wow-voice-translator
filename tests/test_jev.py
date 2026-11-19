"""The Jev decision-model router for the assistant (no network, no models)."""

import json

import httpx
import pytest

from wvt.jev import JevChain, JevClient, JevError, _endpoint, jev_router

URL = "https://opencode.ai/inference/systemone/v1/systemone"
CRITERIA = {"plan": "what to do next", "other": "ordinary conversation or a general game-knowledge question"}


def _client(handler, **kw):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return JevClient("secret-key-123", client=client, **kw), seen


def _answers(is_command=True, choice="plan", confidence=0.82):
    return {"answers": {"is_command": {"type": "noul", "noul": is_command},
                        "intent": {"type": "choice", "choice": choice, "confidence": confidence,
                                   "probabilities": {choice: confidence, "other": round(1 - confidence, 2)}}},
            "usage": {"input_tokens": 40, "output_tokens": 5}}


def test_request_url_headers_and_body():
    j, seen = _client(lambda r: httpx.Response(200, json=_answers()))
    decision = j.classify("co teraz?", CRITERIA)
    assert decision is not None
    assert decision.label == "plan" and decision.is_command is True
    assert decision.confidence == pytest.approx(0.82)
    assert decision.probabilities["plan"] == pytest.approx(0.82)
    assert decision.usage["output_tokens"] == 5 and decision.ms >= 0
    req = seen[0]
    assert str(req.url) == URL
    assert req.headers["Authorization"] == "Bearer secret-key-123"
    assert req.headers["Content-Type"] == "application/json"
    body = json.loads(req.content)
    assert body["model"] == "jev-1.13" and body["state"] == "co teraz?"
    questions = body["questions"]
    assert questions["is_command"]["type"] == "noul" and questions["is_command"]["instructions"]
    assert questions["intent"]["type"] == "choice"
    assert questions["intent"]["criteria"] == CRITERIA
    assert "secret-key-123" not in str(req.url)


def test_a_base_url_without_systemone_gets_it_and_a_trailing_slash_is_dropped():
    j, seen = _client(lambda r: httpx.Response(200, json=_answers()), base_url="https://opencode.ai/inference/systemone/v1/")
    j.classify("co teraz?", CRITERIA)
    assert j.url == URL and str(seen[0].url) == URL


def test_endpoint_rules():
    assert _endpoint("https://opencode.ai/inference/systemone/v1/systemone") == \
        "https://opencode.ai/inference/systemone/v1/systemone"
    assert _endpoint("https://openrouter.ai/api/alpha/decisions") == \
        "https://openrouter.ai/api/alpha/decisions"
    assert _endpoint("https://opencode.ai/inference/systemone/v1") == \
        "https://opencode.ai/inference/systemone/v1/systemone"
    assert _endpoint("https://example.test/api") == "https://example.test/api/systemone"
    assert _endpoint("https://example.test/api/v1") == "https://example.test/api/v1/systemone"
    assert _endpoint("https://example.test/base") == "https://example.test/base/systemone"


def test_decision_carries_the_provider():
    j, _ = _client(lambda r: httpx.Response(200, json=_answers()), provider="openrouter")
    decision = j.classify("co teraz?", CRITERIA)
    assert decision is not None and decision.provider == "openrouter"
    j2, _ = _client(lambda r: httpx.Response(200, json=_answers()))
    assert j2.classify("co teraz?", CRITERIA).provider == "jev-1.13"  # provider falls back to the model


def test_not_a_command_is_a_chat_decision():
    j, _ = _client(lambda r: httpx.Response(200, json=_answers(is_command=False)))
    decision = j.classify("kim był Arthas?", CRITERIA)
    assert decision is not None
    assert decision.label == "chat" and decision.is_command is False
    assert decision.confidence == 1.0 and decision.probabilities == {}


def test_low_confidence_or_other_is_no_decision():
    for answer in (_answers(confidence=0.2), _answers(choice="other", confidence=0.99)):
        j, _ = _client(lambda r, a=answer: httpx.Response(200, json=a))
        assert j.classify("co teraz?", CRITERIA) is None


def test_a_limit_rests_the_router_and_raises():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"detail": "Rate limit reached"}, headers={"Retry-After": "7"})

    j, seen = _client(handler)
    with pytest.raises(JevError, match="Rate limit"):
        j.classify("co teraz?", CRITERIA)
    assert not j.available() and j.failures == 1 and len(seen) == 1
    assert j.classify("co teraz?", CRITERIA) is None  # resting: no second network call
    assert len(seen) == 1 and len(calls) == 1


def test_a_non_json_body_is_an_unreadable_answer():
    j, _ = _client(lambda r: httpx.Response(200, text="<html>bad gateway</html>"))
    with pytest.raises(JevError, match="unreadable answer"):
        j.classify("co teraz?", CRITERIA)
    assert j.failures == 1


class Cfg(dict):
    def get(self, dotted, default=None):
        return super().get(dotted, default)


def test_jev_router_needs_the_switch_and_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    client = jev_router(Cfg({"assistant.router": "jev"}))
    assert isinstance(client, JevChain) and client.model_name == "jev-1.13 (opencode)"
    client = jev_router(Cfg({"assistant.router": "jev", "assistant.jev": {"model": "jev-1.13-free"}}))
    assert isinstance(client, JevChain) and client.model_name == "jev-1.13-free (opencode)"
    assert jev_router(Cfg({})) is None  # default router: local
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    assert jev_router(Cfg({"assistant.router": "jev"})) is None


def test_jev_router_orders_openrouter_then_opencode(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    router = jev_router(Cfg({"assistant.router": "jev",
                             "assistant.jev": {"order": ["openrouter", "opencode"]}}))
    assert isinstance(router, JevChain)
    assert "typesafe/jev-1.14 (openrouter)" in router.model_name
    assert "jev-1.13 (opencode)" in router.model_name
    assert router.model_name.index("openrouter") < router.model_name.index("opencode")
    router.close()


def test_jev_router_falls_back_when_the_openrouter_key_is_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    router = jev_router(Cfg({"assistant.router": "jev",
                             "assistant.jev": {"order": ["openrouter", "opencode"]}}))
    assert isinstance(router, JevChain) and router.model_name == "jev-1.13 (opencode)"
    router.close()


def test_jev_router_order_opencode_only_disables_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    router = jev_router(Cfg({"assistant.router": "jev", "assistant.jev": {"order": ["opencode"]}}))
    assert isinstance(router, JevChain) and router.model_name == "jev-1.13 (opencode)"
    router.close()


class _FakeClient:
    def __init__(self, model="m", provider="p", decision=None, error=None, available=True):
        self.model_name = model
        self.provider = provider
        self.decision = decision
        self.error = error
        self._available = available
        self.calls = 0
        self.closed = False

    def available(self):
        return self._available

    def classify(self, text, criteria, instructions=None, min_confidence=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.decision

    def close(self):
        self.closed = True


def _decision(provider="p"):
    from wvt.jev import JevDecision

    return JevDecision("plan", 0.9, {"plan": 0.9}, True, 5, {}, provider)


def test_chain_primary_error_uses_secondary():
    primary = _FakeClient("primary", "openrouter", error=JevError(429, "limit"))
    secondary = _FakeClient("secondary", "opencode", decision=_decision("opencode"))
    chain = JevChain([primary, secondary])
    assert chain.classify("co teraz?", CRITERIA).provider == "opencode"
    assert primary.calls == 1 and secondary.calls == 1


def test_chain_primary_none_uses_secondary():
    primary = _FakeClient("primary", "openrouter", decision=None)
    secondary = _FakeClient("secondary", "opencode", decision=_decision("opencode"))
    chain = JevChain([primary, secondary])
    assert chain.classify("co teraz?", CRITERIA).provider == "opencode"
    assert secondary.calls == 1


def test_chain_primary_decision_skips_secondary():
    primary = _FakeClient("primary", "openrouter", decision=_decision("openrouter"))
    secondary = _FakeClient("secondary", "opencode", decision=_decision("opencode"))
    chain = JevChain([primary, secondary])
    assert chain.classify("co teraz?", CRITERIA).provider == "openrouter"
    assert secondary.calls == 0


def test_chain_unavailable_primary_uses_secondary():
    primary = _FakeClient("primary", "openrouter", decision=_decision("openrouter"), available=False)
    secondary = _FakeClient("secondary", "opencode", decision=_decision("opencode"))
    chain = JevChain([primary, secondary])
    assert chain.classify("co teraz?", CRITERIA).provider == "opencode"
    assert primary.calls == 0 and secondary.calls == 1


def test_chain_all_fail_is_none():
    chain = JevChain([_FakeClient("a", "openrouter", error=JevError(0, "boom")),
                      _FakeClient("b", "opencode", error=RuntimeError("nope"))])
    assert chain.classify("co teraz?", CRITERIA) is None


def test_chain_model_name_and_close():
    a = _FakeClient("typesafe/jev-1.14", "openrouter")
    b = _FakeClient("jev-1.13", "opencode")
    chain = JevChain([a, None, b])
    assert chain.model_name == "typesafe/jev-1.14 (openrouter) -> jev-1.13 (opencode)"
    assert chain.available()
    chain.close()
    assert a.closed and b.closed

