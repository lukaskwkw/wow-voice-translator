"""The OpenAI-compatible cloud model for the assistant (no network, no models)."""

import json

import httpx
import pytest

from wvt.gemini import FallbackLLM, assistant_llm
from wvt.openai_compat import OpenAICompatError, OpenAICompatLLM

BASE = "https://cloud.example/v1"


def _llm(handler, base_url=BASE, **kw):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return OpenAICompatLLM("secret-key-123", "deepseek-v4.1-flash", base_url, client=client, **kw), seen


def _answer(text="Hogger stoi w Elwynn.", finish="stop"):
    return {"model": "deepseek-v4.1-flash", "choices": [{"message": {"role": "assistant", "content": text},
                                                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}}


def test_request_url_headers_and_body():
    llm, seen = _llm(lambda r: httpx.Response(200, json=_answer()), extra_headers={"X-Title": "wvt"})
    messages = [{"role": "system", "content": "Iskra"}, {"role": "user", "content": "Gdzie Hogger?"}]
    text, info = llm.chat(messages, max_tokens=120, temperature=0.4, stop=["\n\n"])
    assert text == "Hogger stoi w Elwynn." and info["finish"] == "stop"
    assert info["usage"]["completion_tokens"] == 7 and info["model"] == "deepseek-v4.1-flash"
    req = seen[0]
    assert str(req.url) == BASE + "/chat/completions"
    assert req.headers["Authorization"] == "Bearer secret-key-123"
    assert req.headers["Content-Type"] == "application/json" and req.headers["X-Title"] == "wvt"
    assert "secret-key-123" not in str(req.url)
    body = json.loads(req.content)
    assert body == {"model": "deepseek-v4.1-flash", "messages": messages, "stream": False,
                    "max_tokens": 120, "temperature": 0.4, "stop": ["\n\n"]}


def test_a_content_list_is_joined_and_a_trailing_slash_dropped():
    answer = {"choices": [{"message": {"content": [{"type": "text", "text": "Cześć"}, {"type": "text", "text": "!"}]},
                           "finish_reason": "stop"}]}
    llm, seen = _llm(lambda r: httpx.Response(200, json=answer), base_url=BASE + "/")
    assert llm.chat([{"role": "user", "content": "hej"}], max_tokens=0)[0] == "Cześć!"
    assert str(seen[0].url) == BASE + "/chat/completions"
    assert json.loads(seen[0].content)["max_tokens"] == llm.max_output_tokens  # max_tokens 0 -> the configured cap


def test_a_base_url_ending_in_chat_completions_is_not_doubled():
    llm, seen = _llm(lambda r: httpx.Response(200, json=_answer()), base_url="https://cloud.example/v1/chat/completions")
    llm.chat([{"role": "user", "content": "hej"}])
    assert str(seen[0].url) == "https://cloud.example/v1/chat/completions"
    assert llm.model_name == "deepseek-v4.1-flash" and llm.supports_thinking is False


def test_an_empty_answer_raises():
    llm, _ = _llm(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "  "},
                                                                   "finish_reason": "length"}]}))
    with pytest.raises(OpenAICompatError, match="length"):
        llm.chat([{"role": "user", "content": "hej"}])
    assert llm.failures == 1


def test_a_limit_makes_the_cloud_rest_and_raise():
    llm, seen = _llm(lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached for requests"}}))
    with pytest.raises(OpenAICompatError, match="Rate limit"):
        llm.chat([{"role": "user", "content": "hej"}])
    assert not llm.available() and llm.failures == 1
    assert len(seen) == 1


class FakeLocal:
    model_name = "Nemotron"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=200, temperature=0.0, stop=None, timeout=None):
        self.calls += 1
        return "lokalnie", {}


def test_a_non_json_body_is_an_unreadable_answer():
    llm, seen = _llm(lambda r: httpx.Response(200, text="<html>bad gateway</html>"))
    with pytest.raises(OpenAICompatError, match="unreadable answer"):
        llm.chat([{"role": "user", "content": "hej"}])
    assert llm.failures == 1
    local = FakeLocal()
    wrapper = FallbackLLM(llm, local)
    assert wrapper.chat([{"role": "user", "content": "hej"}])[0] == "lokalnie"
    assert local.calls == 1 and wrapper.fallbacks == 1 and len(seen) == 2


def test_a_json_list_body_is_an_unreadable_answer():
    llm, seen = _llm(lambda r: httpx.Response(200, json=[1, 2, 3]))
    with pytest.raises(OpenAICompatError, match="unreadable answer"):
        llm.chat([{"role": "user", "content": "hej"}])
    assert llm.failures == 1
    local = FakeLocal()
    wrapper = FallbackLLM(llm, local)
    assert wrapper.chat([{"role": "user", "content": "hej"}])[0] == "lokalnie"
    assert local.calls == 1 and wrapper.fallbacks == 1 and len(seen) == 2


def test_defaults_carry_the_openai_section_and_confirm_off():
    from wvt.config import DEFAULTS

    o = DEFAULTS["assistant"]["openai"]
    for key in ("base_url", "api_key_env", "model"):
        assert key in o
    assert DEFAULTS["insert"]["confirm"] is False


def test_fallback_uses_the_local_model_on_error():
    llm, seen = _llm(lambda r: httpx.Response(500, json={"error": {"message": "boom"}}))
    local = FakeLocal()
    wrapper = FallbackLLM(llm, local)
    msgs = [{"role": "user", "content": "Hej"}]
    assert wrapper.chat(msgs)[0] == "lokalnie" and len(seen) == 1
    assert wrapper.chat(msgs)[0] == "lokalnie" and len(seen) == 1  # resting: not asked again at once
    assert local.calls == 2 and wrapper.fallbacks == 2
    assert wrapper.supports_thinking is False and wrapper.model_name == "deepseek-v4.1-flash (zapas: Nemotron)"


class Cfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


def test_assistant_llm_openai_needs_a_key_and_wraps_the_cloud_model(monkeypatch):
    local = FakeLocal()
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    assert assistant_llm(Cfg({"assistant.llm": "openai"}), local) is local  # no key: stays local
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    chosen = assistant_llm(Cfg({"assistant.llm": "openai",
                                "assistant.openai": {"model": "deepseek-v4.1-flash"}}), local)
    assert isinstance(chosen, FallbackLLM) and "deepseek-v4.1-flash" in chosen.model_name
    assert chosen.model_name.endswith("(zapas: Nemotron)")
