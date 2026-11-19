"""Jev labels mapped to intents, and the router lookup in the assistant (no network)."""

from wvt.assistant import Assistant, intent_from_label
from wvt.jev import JevDecision


def test_plan_maps_to_plan_next():
    assert intent_from_label("plan", "co teraz?").kind == "plan_next"


def test_summary_keeps_the_channel():
    assert intent_from_label("summary", "co piszą na trade?").channel == "trade"
    assert intent_from_label("summary", "streszcz czat").channel == ""


def test_invite_prefers_a_known_name():
    intent = intent_from_label("invite", "dopchnij Artasa", names=("Artas",))
    assert intent.kind == "invite_name" and intent.spoken == "Artas"


def test_invite_with_a_pronoun_is_the_last_player():
    assert intent_from_label("invite", "weź go do party").kind == "invite_last"


def test_kick_and_leader_need_a_name():
    assert intent_from_label("kick", "wyrzuć kogoś").kind == "chat"
    kicked = intent_from_label("kick", "wyrzuć Artasa", names=("Artas",))
    assert kicked.kind == "kick" and kicked.spoken == "Artas"
    led = intent_from_label("leader", "daj lidera Artasowi", names=("Artas",))
    assert led.kind == "lead" and led.spoken == "Artas"


def test_reminder_and_note_list_vs_add():
    assert intent_from_label("reminder", "jakie mam przypomnienia").kind == "reminder_list"
    assert intent_from_label("reminder", "przypomnij mi za 10 minut o wodzie").kind == "reminder_add"
    assert intent_from_label("note", "pokaż notatki").kind == "note_list"
    assert intent_from_label("note", "zanotuj, że kupić wodę").kind == "note_add"


def test_other_and_unknown_are_chat():
    assert intent_from_label("other", "kim był Arthas?").kind == "chat"
    assert intent_from_label("", "cześć").kind == "chat"
    assert intent_from_label("no_such_label", "cześć").kind == "chat"


class FakeRouter:
    def __init__(self, decision):
        self.decision = decision

    def classify(self, text, criteria, instructions=None, min_confidence=None):
        return self.decision


def _assistant(router):
    a = Assistant.__new__(Assistant)
    a.router = router
    a._known_names = lambda: ["Artas"]
    return a


def test_route_intent_maps_a_command():
    decision = JevDecision("invite", 0.82, {"invite": 0.82}, True, 380)
    intent = _assistant(FakeRouter(decision))._route_intent("dopchnij Artasa")
    assert intent is not None and intent.kind == "invite_name" and intent.spoken == "Artas"


def test_route_intent_ignores_non_commands_and_misses():
    assert _assistant(FakeRouter(None))._route_intent("co teraz?") is None
    chatting = JevDecision("chat", 1.0, {}, False, 300)
    assert _assistant(FakeRouter(chatting))._route_intent("kim był Arthas?") is None


def test_without_a_router_nothing_is_routed():
    assert _assistant(None)._route_intent("co teraz?") is None
