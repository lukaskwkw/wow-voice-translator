"""Follow-up lists ("drugi", "nr 3"), quest difficulty and trainer phrasing for Iskra."""

import time
from types import SimpleNamespace

from tests.test_assistant import make_assistant, quest_pkt
from wvt.advisor import _service, find_place
from wvt.assistant import TOPIC_SECONDS, parse_intent
from wvt.kb.world import Player, World

POS = SimpleNamespace(map_id=1436, x=0.5, y=0.5, width=3500, height=2333)


class GearTwoWorld:
    """Two upgrade quests in the journal, one of them in Loch Modan (for the words of a follow-up)."""

    names = {10: "Westfall", 20: "Loch Modan"}
    dungeons = {}
    all_items = {
        900: {"n": "Soldier's Sword", "c": 2, "sc": 7, "il": 18, "rl": 10},
        901: {"n": "Signet Ring", "c": 4, "sc": 0, "il": 17, "rl": 10},
    }
    quests = {
        501: {"n": "The Deadmines", "lvl": 15, "zone": 10, "rw": [900]},
        502: {"n": "Loch Modan Quest", "lvl": 15, "zone": 10, "rw": [901]},
    }

    def find_area(self, name):
        return 10 if name == "Westfall" else 0

    def zone_of_map(self, map_id):
        return 10 if map_id == 10 else 0

    def map_of(self, area):
        return 10 if area == 10 else 0

    def available(self, q, qid, player, active):
        return True

    def starter_spot(self, qid, pos, faction):
        return SimpleNamespace(name="Gryan Stoutmantle", distance=30.0, area=10, x=50.0, y=50.0, ui_map=10)

    def finisher_spot(self, qid, pos):
        return None

    def area_name(self, area):
        return self.names.get(area, "")


def gear_two_assistant():
    a, *_ = make_assistant()
    a.world_getter = lambda: GearTwoWorld()
    a.state = SimpleNamespace(level=15, zone="Westfall", subzone="", members=[])
    a.quests.player = SimpleNamespace(race_id=1, class_id=1, faction_code="A")
    a.quests.pos = SimpleNamespace(map_id=10, x=0.5, y=0.5, width=1000, height=1000)
    a.quests.updated_at = time.time()
    a.quests.done_known = False
    a._worn_pieces = lambda: [("Item A", 10, 4, 1), ("Item B", 10, 4, 1), ("Item C", 10, 4, 1),
                              ("Item D", 10, 4, 1), ("Item E", 10, 4, 1)]
    for i, (qid, title, zone) in enumerate(((501, "The Deadmines", "Westfall"),
                                            (502, "Loch Modan Quest", "Loch Modan")), start=1):
        a.on_packet(quest_pkt(sync=1, index=i, total=2, quest_id=qid, title=title, zone=zone))
    return a


def test_gear_next_remembers_its_rows_for_ordinals_and_words():
    a = gear_two_assistant()
    a.handle("gdzie iść po upgrade itemów?")
    assert a._last_list is not None and a._last_list.kind == "quest"
    assert [i[0] for i in a._last_list.items] == [501, 502]
    assert a._pick_from_list("a jak trudny jest ten drugi?") == ("quest", 502)
    assert a._pick_from_list("a co z tym z Loch Modan?") == ("quest", 502)
    assert a._pick_from_list("nr 2") == ("quest", 502)
    assert a._pick_from_list("nr 5") is None  # beyond the list
    # the difficulty answer goes through the same list
    text, _line = a._difficulty("a jak trudny jest ten drugi?")
    assert "Loch Modan Quest" in text


def test_the_shown_list_expires_and_the_old_behaviour_returns():
    a = gear_two_assistant()
    a.handle("gdzie iść po upgrade itemów?")
    a._last_list.at = time.time() - TOPIC_SECONDS - 1
    assert a._pick_from_list("a jak trudny jest ten drugi?") is None
    assert a._difficulty("a jak trudny jest ten drugi?") == ("", "")


class DifficultyWorld:
    names = {10: "Westfall", 20: "Loch Modan", 30: "The Deadmines"}
    dungeons = {30: {"n": "The Deadmines", "at": True}}
    quests = {
        601: {"n": "Easy Job", "lvl": 13, "zone": 10, "sn": [700], "en": [701]},
        602: {"n": "Even Job", "lvl": 17, "zone": 10, "sn": [700], "en": [701]},
        603: {"n": "Hard Job", "lvl": 20, "zone": 10, "sn": [700], "en": [701]},
        604: {"n": "Elite Hunt", "lvl": 17, "zone": 10, "kill": [[900, None]], "sn": [700], "en": [701]},
        605: {"n": "Dungeon Job", "lvl": 17, "zone": 30, "sn": [700], "en": [701]},
        606: {"n": "Group Job", "lvl": 17, "zone": 10, "sn": [700], "en": [701]},
    }
    npcs = {
        700: {"n": "Giver", "sp": {"10": [[50.0, 50.0]]}},
        701: {"n": "Finisher", "sp": {"10": [[52.0, 52.0]]}},
        900: {"n": "Elite Mob", "r": 1, "sp": {"10": [[60.0, 60.0]]}},
    }

    def find_area(self, name):
        return 10 if name == "Westfall" else 0

    def zone_of_map(self, map_id):
        return map_id

    def map_of(self, area):
        return area

    def _spot(self, nid, distance):
        n = self.npcs[nid]
        return SimpleNamespace(name=n["n"], distance=distance, area=10, x=50.0, y=50.0, ui_map=10)

    def starter_spot(self, qid, pos, faction):
        q = self.quests.get(qid or 0)
        return self._spot(q["sn"][0], 100.0) if q and q.get("sn") else None

    def finisher_spot(self, qid, pos):
        q = self.quests.get(qid or 0)
        return self._spot(q["en"][0], 120.0) if q and q.get("en") else None


def difficulty_assistant():
    a, *_ = make_assistant()
    a.world_getter = lambda: DifficultyWorld()
    a.state = SimpleNamespace(level=17, zone="Westfall", subzone="", members=[])
    a.quests.pos = SimpleNamespace(map_id=10, x=0.5, y=0.5, width=1000, height=1000)
    a.quests.player = SimpleNamespace(race_id=1, class_id=1, faction_code="A")
    rows = ((601, "Easy Job", 13, "Westfall", 0), (602, "Even Job", 17, "Westfall", 0),
            (603, "Hard Job", 20, "Westfall", 0),
            (605, "Dungeon Job", 17, "The Deadmines", 0), (606, "Group Job", 17, "Westfall", 2))
    for i, (qid, title, lvl, zone, group) in enumerate(rows, start=1):
        a.on_packet(quest_pkt(sync=1, index=i, total=len(rows), quest_id=qid, title=title, level=lvl,
                              zone=zone, group=group))
    return a


def test_difficulty_verdicts_follow_the_level_delta():
    a = difficulty_assistant()
    easy, _ = a._difficulty("Easy Job")
    assert "Easy Job (poz. 13, Ty 17)" in easy and "łatwy" in easy
    even, _ = a._difficulty("Even Job")
    assert "Even Job (poz. 17, Ty 17)" in even and "w sam raz" in even
    hard, _ = a._difficulty("Hard Job")
    assert "Hard Job (poz. 20, Ty 17)" in hard and "trudniejszy niż Twój poziom" in hard


def test_difficulty_reads_the_group_elites_and_the_dungeon():
    a = difficulty_assistant()
    group, _ = a._difficulty("Group Job")
    assert "na grupę 2" in group
    a._remember_list("quest", [(604, "Elite Hunt", "Westfall")])  # a world quest she just listed, not in the log
    elite, _ = a._difficulty("Elite Hunt")
    assert ", elity" in elite
    dungeon, _ = a._difficulty("Dungeon Job")
    assert "prowadzi do The Deadmines" in dungeon


def test_difficulty_without_any_quest_returns_nothing():
    a, *_ = make_assistant()
    a.world_getter = lambda: DifficultyWorld()
    a.state = SimpleNamespace(level=17, zone="Westfall", subzone="", members=[])
    a.quests.player = SimpleNamespace(race_id=1, class_id=1, faction_code="A")
    assert a._difficulty("jak trudny jest ten quest?") == ("", "")


def test_difficulty_routes_before_the_broader_quest_rules():
    assert parse_intent("jak trudny jest ten quest?").kind == "quest_difficulty"
    assert parse_intent("czy dam radę sam?").kind == "quest_difficulty"
    assert parse_intent("jaki dungeon na mój poziom?").kind == "dungeon_level"
    assert parse_intent("czy warto robić ten quest?").kind == "quest_worth"


def test_trainer_phrasing_finds_cooks_and_weapon_masters():
    assert parse_intent("gdzie nauczę się gotować").kind == "find"
    assert parse_intent("gdzie nauczę się rzucać").kind == "spell_trainer"  # the optional gamedb knows the trainers
    me = Player(level=10, class_id=1, faction="A")
    assert _service("gdzie naucze sie gotowac", me)[:2] == ("TRAINER", "Cook")
    assert _service("gdzie naucze sie rzucac", me)[:2] == ("TRAINER", "Weapon Master")


def test_a_weapon_master_answer_says_it_has_no_spell_list():
    world = World({
        "quests": {}, "objects": {}, "items": {},
        "npcs": {1: {"n": "Woo Ping", "sub": "Weapon Master", "fl": 16, "f": "A",
                     "sp": {"40": [[56.0, 47.0]]}}},
        "zones": {"names": {40: "Westfall"}, "map": {40: 1436}},
    })
    adv = find_place(world, "gdzie nauczę się rzucać", Player(level=10, class_id=1, faction="A"), POS)
    assert adv is not None and "Woo Ping" in adv.text
    assert "mistrz broni" in adv.text and "zapytaj go o tę" in adv.text


def test_an_item_question_routed_to_difficulty_falls_back_to_the_item_facts():
    a, *_ = make_assistant()
    a.world_getter = lambda: World({
        "quests": {}, "npcs": {}, "objects": {}, "items": {}, "zones": {},
        "all_items": {900: {"n": "Oily Blackmouth", "c": 5, "il": 15, "rl": 0}},
    })
    a.state = SimpleNamespace(level=17, zone="Westfall", subzone="", members=[])
    a.quests.player = SimpleNamespace(race_id=1, class_id=1, faction_code="A")
    reply, _line = a.handle("czy trudny jest ten Oily Blackmouth do zdobycia?")
    assert a.transcript[0]["intent"] == "quest_difficulty"  # the rule still wins the routing
    assert "Oily Blackmouth" in reply and "Brak danych o cenach rynkowych." in reply
    assert "Ty 17" not in reply and "w sam raz" not in reply


def test_a_pure_difficulty_question_still_answers_about_the_quest():
    a = difficulty_assistant()
    a._set_topic("quest", "Easy Job", 601)  # the quest she is talking about
    reply, _line = a.handle("jak trudne jest to zadanie?")
    assert "Easy Job (poz. 13, Ty 17)" in reply and "łatwy" in reply
