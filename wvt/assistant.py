"""Voice assistant: her own push-to-talk key, Polish speech in, a spoken answer and game commands out.

She ("Iskra" by default, ``assistant.name``) gets what you said as Polish text from the push-to-talk
pipeline and:

* **group** - invites everybody who asked for an invite ("inv me", "inv pls", "can I get an invite"
  in /say or /yell, or "inv"/"+" in a whisper, during the last 5 minutes) - up to the free party
  slots; invites, kicks or promotes a named player; makes a raid; leaves; starts a ready check.
  These become ONE line in the chat box ("/wvt inv Ann Bob") that runs only when you press Enter -
  nothing is ever sent by itself.  Kick, lead, raid and ready check are refused by the game during
  combat, boss fights, M+ and PvP matches (Midnight addon restrictions) - she says so;
* **chat** - who asked for an invite, what the last whispers said (in Polish), what people are
  talking about (say, party, guild, Trade, General, who is looking for a group), a reply to a whisper
  or a line for /p, /s, /g, /ra (translated to English, waiting in the chat box);
* **items** - "gdzie kupię / skąd wezmę Rhapsody Malt?": who sells it, what drops it, where the
  nearest one is, with a map waypoint (QuestieDB items of quests);
* **your loot** - what you picked up in the last minutes, and a word when a looted item beats what
  you wear;
* **your bags** - "czy ten kij, który znalazłem, jest lepszy?" (the item is found by its kind, colour,
  name or the small vector index of the names of the moment, and compared with what you wear), "co mam
  w plecaku?", "co mogę sprzedać?", "czy mam Rhapsody Malt?";
* **you** - gold, level and XP to the next level, where you are, free bag slots, who is in the
  group (from the addon's state packet);
* **voices** - mute or unmute the character voices, Polish/English voice mode, /say and /party
  voices on or off;
* **quests** - what is in your quest log, how far you are with a quest, what you can turn in,
  where to go (direction and distance to the game's own map marker of the next objective), which
  quest is nearest; free questions about your quests go to the LLM with the quest log as facts;
* **where next** - what to do now (turn-ins, the nearest open quest, new quests to pick up near you
  for your level, class, faction and completed quests, or the zone that fits your level), which
  dungeon fits your level, where the nearest trainer / innkeeper / flight master / an NPC is - with
  a map waypoint ready in the chat box (``/wvt way``, ``/wvt track``);
* **reminders** - "przypomnij mi za 10 minut / o 21:30 / na 20 poziomie, że..." (wvt/reminders.py);
* **session** - XP and gold per hour, time to the next level (wvt/session_stats.py);
* **loot** - when a group loot roll starts she says whether it is a need or a greed for your class,
  level and worn gear (wvt/loot.py);
* **guides** - questions about the game answered from the local knowledge base (Warcraft Wiki pages
  for WoW Forever/Classic, QuestieDB quests, your notes; wvt/kb) by the LLM, a few concrete sentences;
* anything else - a short chat in character (the local LLM; she may be wrong about game facts).

With ``privacy.log_assistant`` every question, what she understood (intent), her answer and its
sources go to ``assistant.log_file`` (JSON lines), so wrong answers can be found and fixed.

Commands are matched by rules on the Polish text (fast, predictable, nothing invented); the LLM only
talks and summarises.  Before that, names Whisper misheard ("Targasa", "gnome Regan", "Kinaj") are
replaced by the game's spelling from the names of the moment (wvt/lexicon.py: your quests, the people
and places around you, your loot and bags), and her LLM gets a short picture of your game.
"""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import logging
import random
import re
import threading
import time
import unicodedata
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import protocol
from .advisor import Advice, CLASS_EN, dungeons_for_level, find_item, find_place, spot_where, where_next
from .bags import BagBook
from .chat_insert import build_chat_line
from .config import resolve_path
from .directives import DIRECTIVES, is_real_question, split as split_directives
from .jev import JevDecision, jev_router
from .kb.world import Player
from .lexicon import Correction, Lexicon
from .loot import advise as loot_advise, best_armor, item_verdict, pickup_tip, SHIELD_CLASSES, WEAPONS
from .planner import Plan, Step, make_plan, skip_sentence
from .quests import QuestBook, match_quests, spoken_objective
from .quest_tips import Chains, assess
from .reminders import (Reminder, Reminders, describe as describe_reminder, parse_note, parse_reminder,
                        say_reminder)
from .session_stats import SessionStats

log = logging.getLogger(__name__)

INVITE_KINDS = ("SAY", "YELL", "WHISPER")
TALK_KINDS = ("SAY", "YELL", "PARTY", "PARTY_LEADER", "RAID", "RAID_LEADER", "WHISPER", "WHISPER_INFORM")
# what "co piszą na czacie?" reads: talk plus raid warnings, guild, instance chat and public channels
READ_KINDS = TALK_KINDS + ("RAID_WARNING", "GUILD", "OFFICER", "INSTANCE_CHAT", "INSTANCE_CHAT_LEADER", "CHANNEL")
LOOT_KINDS = ("LOOT", "MONEY")  # the player's own loot lines
NPC_KINDS = ("MONSTER_SAY", "MONSTER_YELL", "MONSTER_WHISPER")  # NPC speech: names for the lexicon only
CHANNEL_CMDS = {"party": "/p", "say": "/s", "guild": "/g", "raid": "/ra", "yell": "/y"}


# --------------------------------------------------------------------------- text helpers

def fold(text: str) -> str:
    """Lower case, no Polish diacritics, no punctuation, single spaces - for matching what was heard."""
    t = (text or "").lower().replace("ł", "l")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^\w\s+]", " ", t)
    return " ".join(t.split())


def _words_with_original(text: str) -> tuple[list[str], list[str]]:
    """The same words twice: folded for matching and as spoken (diacritics kept) for messages."""
    original = re.sub(r"[^\w\s+]", " ", text or "").split()
    return [fold(w) for w in original], original


def name_key(name: str) -> str:
    return fold((name or "").split("-")[0])


def _phonetic(s: str) -> str:
    s = fold(s)
    for a, b in (("th", "t"), ("ph", "f"), ("ck", "k"), ("w", "v"), ("y", "i"), ("ee", "i"), ("oo", "u"),
                 ("sz", "sh"), ("cz", "ch"), ("j", "i")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


# Polish case endings speech recognition writes on foreign names: Arthasa, Arthasowi, Arthasem, Jainie, Jainę...
_ENDINGS = ("owi", "ami", "em", "ie", "om", "a", "e", "u", "y", "i", "o")


def match_name(spoken: str, names: list[str], cutoff: float = 0.72) -> Optional[str]:
    """The known character name that the spoken words most likely mean, or None."""
    best, best_score = None, 0.0
    words = [w for w in fold(spoken).split() if len(w) >= 3]
    for name in names:
        base = _phonetic(name.split("-")[0])
        if len(base) < 2:
            continue
        for w in words:
            pw = _phonetic(w)
            variants = {pw} | {pw[: -len(e)] for e in _ENDINGS if pw.endswith(e) and len(pw) - len(e) >= 3}
            for v in variants:
                score = difflib.SequenceMatcher(None, v, base).ratio()
                if score > best_score:
                    best, best_score = name, score
    return best if best_score >= cutoff else None


def polish_list(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " i " + items[-1]


def plural(n: int, one: str, few: str, many: str) -> str:
    if n == 1:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# --------------------------------------------------------------------------- invite requests

_INVITE_PATTERNS = (
    r"^(?:pl[sz]|please|pls|plz)?\s*(?:inv|invite|inw|invi)(?:\s+(?:me|pls|plz|please|pl[sz]|ty|thx|pl0x|mnie))*$",
    r"\binv(?:ite)?\s+(?:me|pls|plz|please)\b",
    r"\b(?:can|could|may)\s+i\s+(?:get\s+)?(?:an?\s+)?inv(?:ite)?\b",
    r"\b(?:inv|invite)\s+(?:me\s+)?to\s+(?:the\s+|ur\s+|your\s+)?(?:group|party|grp|raid)\b",
    r"\bzapro[sś]\w*\s+mnie\b",
)
_INVITE_RE = re.compile("|".join(f"(?:{p})" for p in _INVITE_PATTERNS))
_WHISPER_INVITE_RE = re.compile(r"^(?:\+|1|inv\w*|invite\w*)$")


def is_invite_request(text: str, kind: str) -> bool:
    t = " ".join((text or "").lower().split()).strip(" !.?")
    if not t or len(t) > 80:
        return False
    if _INVITE_RE.search(t):
        return True
    return kind == "WHISPER" and bool(_WHISPER_INVITE_RE.match(t))


# --------------------------------------------------------------------------- chat log

@dataclass
class ChatLine:
    when: float
    kind: str
    sender: str  # for WHISPER_INFORM: the player you whispered to
    text: str
    is_self: bool
    channel: str = ""  # public channels: "Trade", "General", "LookingForGroup"...


class ChatLog:
    """Recent chat lines from the addon (spoken or not).  Kept in memory only, never written to disk.

    Public channels and the player's loot lines have deques of their own, so a busy Trade chat or a
    long farming session never pushes the whispers and invite requests out."""

    def __init__(self, max_lines: int = 300):
        self.lines: deque[ChatLine] = deque(maxlen=max_lines)
        self.channel_lines: deque[ChatLine] = deque(maxlen=200)
        self.loot_lines: deque[ChatLine] = deque(maxlen=120)
        self.npc_lines: deque[ChatLine] = deque(maxlen=60)
        self._lock = threading.Lock()

    def add_packet(self, pkt: protocol.Packet, now: Optional[float] = None) -> None:
        if pkt.type not in (protocol.TYPE_VOICE, protocol.TYPE_CHAT) or pkt.is_secret or not (pkt.text or "").strip():
            return
        kind = pkt.chat_kind_name
        if kind == "BAG":
            return  # the bag contents (wvt.bags), not chat
        line = ChatLine(now if now is not None else time.time(), kind, pkt.sender, pkt.text.strip(), pkt.is_self,
                        pkt.race if kind == "CHANNEL" else "")  # the addon names the channel in the race field
        with self._lock:
            target = self.channel_lines if kind == "CHANNEL" else self.loot_lines if kind in LOOT_KINDS else \
                self.npc_lines if kind in NPC_KINDS else self.lines
            target.append(line)

    def _snapshot(self, kinds: tuple[str, ...] = ()) -> list[ChatLine]:
        with self._lock:
            out = list(self.lines)
            extra = (list(self.channel_lines) if "CHANNEL" in kinds else []) + \
                (list(self.loot_lines) if any(k in LOOT_KINDS for k in kinds) else []) + \
                (list(self.npc_lines) if any(k in NPC_KINDS for k in kinds) else [])
        if extra:
            out = sorted(out + extra, key=lambda ln: ln.when)
        return out

    def invite_requests(self, window_s: float, exclude: tuple[str, ...] = (), now: Optional[float] = None) -> list[str]:
        """Who asked for an invite (oldest request first), minus ``exclude`` (group members, already invited)."""
        now = now if now is not None else time.time()
        skip = {name_key(n) for n in exclude}
        out: list[str] = []
        for line in self._snapshot():
            if now - line.when > window_s or line.is_self or line.kind not in INVITE_KINDS:
                continue
            key = name_key(line.sender)
            if not key or key in skip or not is_invite_request(line.text, line.kind):
                continue
            skip.add(key)
            out.append(line.sender)
        return out

    def last_speaker(self, window_s: float, exclude: tuple[str, ...] = (), now: Optional[float] = None) -> Optional[str]:
        """The other player who wrote near you or to you last (say, yell, whisper), minus ``exclude``."""
        now = now if now is not None else time.time()
        skip = {name_key(n) for n in exclude}
        for line in reversed(self._snapshot()):
            if now - line.when > window_s:
                break
            key = name_key(line.sender)
            if not line.is_self and line.kind in INVITE_KINDS and key and key not in skip:
                return line.sender
        return None

    def whispers(self, n: int = 3, max_age_s: float = 900, now: Optional[float] = None) -> list[ChatLine]:
        now = now if now is not None else time.time()
        got = [ln for ln in self._snapshot() if ln.kind == "WHISPER" and now - ln.when <= max_age_s]
        return got[-n:]

    def last_whisper_partner(self) -> Optional[str]:
        for line in reversed(self._snapshot()):
            if line.kind in ("WHISPER", "WHISPER_INFORM"):
                return line.sender
        return None

    def recent(self, n: int = 25, kinds: tuple[str, ...] = TALK_KINDS, max_age_s: float = 600,
               now: Optional[float] = None) -> list[ChatLine]:
        now = now if now is not None else time.time()
        got = [ln for ln in self._snapshot(kinds) if ln.kind in kinds and now - ln.when <= max_age_s]
        return got[-n:]

    def names(self) -> list[str]:
        """Everybody seen in chat, most recent first (for matching spoken names)."""
        seen: dict[str, str] = {}
        for line in reversed(self._snapshot()):
            seen.setdefault(name_key(line.sender), line.sender)
        return [n for n in seen.values() if n]


# --------------------------------------------------------------------------- intents

@dataclass
class Intent:
    kind: str
    spoken: str = ""  # the words that should name a player
    message: str = ""  # text to send (as spoken, Polish)
    channel: str = ""  # whisper | party | say | guild | raid | yell
    on: Optional[bool] = None


_GROUP_WORDS = r"(?:ich|wszystkich|chetnych|chetni|tych|tamtych|ludzi|osob\w*|kolesi|gosci|graczy|ekipe|reszte)"
_INVITE_VERB = (r"(?:\w*[sz]apro\w*|\w*[sz]apra\w*|dopros\w*|doprasz\w*|\w*inwajt\w*|\w*invajt\w*|inv|inw"
                r"|invite)")  # zaproś, zapraszaj, doproś (not "doprowadź"), inwajtuj...
# "weź go do party", "wciągnij Ann do grupy", "dorzuć ich do raidu": a verb of taking and where to
_INVITE_PULL = r"(?:wez\w*|wezm\w*|wciagn\w*|wrzuc\w*|dorzuc\w*|dobierz\w*|przyjm\w*|zabierz\w*|sciagn\w*)"
_INVITE_WHERE = r"do (?:party|grup\w*|druzyn\w*|raid\w*|rajd\w*)"
# one person named only by a pronoun ("weź go doproś do party"): the one who asked or spoke last
_ONE_PERSON = {"go", "jego", "niego", "ja", "nia", "jej", "tego", "tamtego", "ta", "te", "tamta", "tamtej",
               "kolesia", "goscia", "typa", "laske", "ziomka", "ziomala", "gracza", "kolege", "kolezanke", "postac"}
# a chat channel and a question about it ("co piszą na trade", "przeczytaj czat", "co nowego na gildii")
_CHAN = (r"(?:czat\w*|chat\w*|trade\w*|tred\w*|tradzie|handlu|handel|kanal\w*|lfg|lfm|gildi\w*|guild\w*"
         r"|na general\w*|na ogolnym)")
_ASK = (r"(?:co|pisz\w*|pisal\w*|mow\w*|gada\w*|dzieje|slychac|czytaj\w*|przeczytaj\w*|stresc\w*|streszcz\w*"
        r"|podsumuj\w*|nowego)")
# which part of the chat a summary is about
_FOCUS = (
    ("lfg", r"\b(lfg|lfm|szuka\w*)\b"),
    ("trade", r"\b(trade\w*|tred\w*|tradzie|handlu|handel|handlow\w*)\b"),
    ("general", r"\b(general\w*|ogoln\w*)\b"),
    ("guild", r"\b(gildi\w*|guild\w*)\b"),
    ("party", r"\b(party|druzyn\w*|grupie|raid\w*|rajd\w*)\b"),
    ("say", r"\b(say\w*|sej\w*|obok|wokol)\b"),
)


def _chat_focus(folded: str) -> str:
    return next((name for name, rx in _FOCUS if re.search(rx, folded)), "")


# "gdzie kupię / skąd wezmę / co to jest <item>": answered from the item data when an item is named
_ITEM_INTENTS = ("quest_where", "find", "chat", "quest_info", "quest_progress", "waypoint")
_ITEM_ASK = re.compile(r"\b(gdzie|skad|kto|kup\w*|sprzeda\w*|zdobyc|zdobed\w*|dostac|dostane|znajde|znalezc|szukac"
                       r"|szukam|wypada\w*|dropi\w*|farmi\w*|zebrac|zbierac|nazbierac|zaprowadz\w*|prowadz\w*"
                       r"|znacznik\w*|co to|czym jest)\b")
# "czy ten kij, który znalazłem, jest lepszy?": a kind of item and a question about it (not a quest, not shopping)
_ITEM_NOUN = (r"(?:kij\w*|kostur\w*|lask\w*|miecz\w*|topor\w*|siekier\w*|sztylet\w*|noz|noza|nozem|nozyk\w*|luk|luku"
              r"|lukiem|kusz\w*|strzelb\w*|rusznic\w*|rozdzk\w*|tarcz\w*|helm\w*|czapk\w*|kaptur\w*|napiersnik\w*"
              r"|zbroj\w*|szat\w*|kurtk\w*|tunik\w*|spodni\w*|nogawic\w*|portk\w*|but|buty|butow|butach|rekawic\w*"
              r"|karwasz\w*|bransolet\w*|pas|pasa|pasek|paska|pierscien\w*|naszyjnik\w*|amulet\w*|plaszcz\w*"
              r"|peleryn\w*|naramiennik\w*|bron\w*|item\w*|przedmiot\w*|rzecz\w*|buzdygan\w*|maczug\w*|bulaw\w*"
              r"|mlot\w*|talizman\w*|blyskotk\w*|pancerz\w*|ryb\w*|material\w*|odczynnik\w*|skladnik\w*|reagent\w*"
              r"|przepis\w*|jedzenie|napoj\w*|zio[łl]\w*|kwiat\w*|rud[ay]\w*|sztab\w*|tkanin\w*|skor\w*|sprzet\w*)")
_ITEM_Q = (r"(?:lepsz\w*|gorsz\w*|slabsz\w*|mocniejsz\w*|warto|zaloz\w*|zakladac|ubrac|dobr\w*|dla mnie|sprzeda\w*"
           r"|do czego|co to|co z|uzyw\w*|nosic|oplaca\w*|ulepszeni\w*|upgrade\w*|przyda\w*)")
# asks the LLM never to guess recipe, price or use facts (assistant.item_help)
_ITEM_COMPARE_RE = re.compile(r"\b(lepsz\w*|gorsz\w*|wart\w*|wiecej|mniej|porown\w*|zamiast|oplaca\w*)\b")
_ITEM_QUESTION_RE = re.compile(
    r"\b(warto|warty|warta|lepsz\w*|gorsz\w*|co (z|moge|mozna|zrobic)|do czego|sprzeda\w*|zachow\w*|co to|czym jest"
    r"|gdzie (kup|zdoby|wypada)|skad|kto sprzedaje|jak zdobyc|gdzie wypada)\b")
_QUEST_WORD_RE = re.compile(r"\b(quest\w*|zadani\w*|misj\w*|dziennik\w*)\b")
# kinds of items by their Polish names: (words, class, subclasses, equip locations)
_ITEM_KINDS = (
    (r"\b(kij\w*|kostur\w*|lask\w*)\b", 2, {10}, ()),
    (r"\b(miecz\w*)\b", 2, {7, 8}, ()),
    (r"\b(topor\w*|siekier\w*)\b", 2, {0, 1}, ()),
    (r"\b(sztylet\w*|noz|noza|nozem|nozyk\w*)\b", 2, {15}, ()),
    (r"\b(luk|luku|lukiem|kusz\w*)\b", 2, {2, 18}, ()),
    (r"\b(strzelb\w*|rusznic\w*)\b", 2, {3}, ()),
    (r"\b(rozdzk\w*)\b", 2, {19}, ()),
    (r"\b(buzdygan\w*|maczug\w*|bulaw\w*|mlot\w*)\b", 2, {4, 5}, ()),
    (r"\b(tarcz\w*)\b", 4, set(), ("INVTYPE_SHIELD",)),
    (r"\b(helm\w*|czapk\w*|kaptur\w*)\b", 4, set(), ("INVTYPE_HEAD",)),
    (r"\b(napiersnik\w*|zbroj\w*|pancerz\w*|szat\w*|kurtk\w*|tunik\w*)\b", 4, set(), ("INVTYPE_CHEST", "INVTYPE_ROBE")),
    (r"\b(spodni\w*|nogawic\w*|portk\w*)\b", 4, set(), ("INVTYPE_LEGS",)),
    (r"\b(but|buty|butow|butach)\b", 4, set(), ("INVTYPE_FEET",)),
    (r"\b(rekawic\w*)\b", 4, set(), ("INVTYPE_HAND",)),
    (r"\b(karwasz\w*|bransolet\w*)\b", 4, set(), ("INVTYPE_WRIST",)),
    (r"\b(pas|pasa|pasek|paska)\b", 4, set(), ("INVTYPE_WAIST",)),
    (r"\b(pierscien\w*)\b", 4, set(), ("INVTYPE_FINGER",)),
    (r"\b(naszyjnik\w*|amulet\w*)\b", 4, set(), ("INVTYPE_NECK",)),
    (r"\b(plaszcz\w*|peleryn\w*)\b", 4, set(), ("INVTYPE_CLOAK",)),
    (r"\b(naramiennik\w*)\b", 4, set(), ("INVTYPE_SHOULDER",)),
    (r"\b(talizman\w*|blyskotk\w*)\b", 4, set(), ("INVTYPE_TRINKET",)),
    (r"\b(bron\w*)\b", 2, set(), ()),
)
_QUALITY_WORDS = ((r"\bszar\w*\b", 0), (r"\bbial\w*\b", 1), (r"\bzielon\w*\b", 2), (r"\bniebiesk\w*\b", 3),
                  (r"\b(fiolet\w*|epik\w*|epic\w*)\b", 4))
# a question that points back at what you were just talking about ("a gdzie go oddać?", "ile za niego expa?")
_BACK_RE = re.compile(r"\b(go|niego|nim|jego|ten|tego|tym|temu|tamten\w*|tam|to|tamto|on|ona|ono|jej|ja|niej|nia)\b"
                      r"|^(a|i|no|ok|okej|dobra)\b")
TOPIC_SECONDS = 900  # a topic (and the conversation) is forgotten after 15 quiet minutes
_FILLER = {"ej", "hej", "no", "prosze", "mi", "ty", "teraz", "szybko", "do", "party", "grupy", "grupa", "druzyny",
           "raidu", "rajdu", "go", "ja", "ta", "ten", "tego", "te", "pana", "pani", "gracza", "gracz", "postac",
           "postaci", "kolege", "kolezanke", "typa", "laske", "goscia", "wez", "daj", "przekaz", "oddaj", "zrob",
           "lidera", "lider", "liderem", "leadera", "lead", "prowadzenie", "z", "ze", "i", "na", "dla", "jemu", "jej",
           "mu", "niech", "bedzie", "mozesz", "moglabys", "mozna", "chce", "chcialbym", "chcialabym", "tez",
           "jeszcze", "dawaj", "pls", "plz"}

_RULES: tuple[tuple[str, str], ...] = (
    # "a potem?", "co dalej?", "ok, co teraz?" - the next step of the plan (the whole utterance, nothing else)
    ("plan_next", r"^(?:(?:a|i|no|dobra|ok|okej|super|dzieki|zrobione|gotowe|oddane)\s+)*(?:co|a)?\s*"
                  r"(?:potem|dalej|nastepne|nastepny|kolejne|kolejny|teraz)$"),
    ("help", r"\b(co umiesz|co potrafisz|w czym (mi )?pomozesz|jakie (masz )?komendy)\b|^pomocy?$"),
    ("stop", r"^(stop|cicho|cisza|przestan\w*|zamknij sie|wystarczy|dosc)\b"),
    ("reminder_list", r"\b(jakie|ile|pokaz\w*|wymien\w*|przeczytaj)\b.*\bprzypomnie\w*\b|^przypomnienia$"),
    ("reminder_cancel", r"\b(usun\w*|anuluj\w*|skasuj\w*|wyczysc\w*|odwolaj\w*|kasuj\w*|zapomnij)\b.*\bprzypomn\w*\b"),
    ("reminder_add", r"\bprzypomn\w*\b|\bprzypominaj\w*\b"),
    ("note_list", r"\b(jakie|co|pokaz\w*|przeczytaj|wymien\w*)\b.*\b(notatk\w*|zanotowal\w*|zapisal\w*)\b"),
    ("note_done", r"\b(usun\w*|skasuj\w*|wyczysc\w*|odhacz\w*|wykresl\w*)\b.*\bnotatk\w*\b|^(zrobione|zalatwione|gotowe)\b"),
    ("note_add", r"^(?:(?:ej|hej|iskra|prosze|i)\s+)*(zanotuj|zapisz|zapamietaj|notatka|dodaj notatke)\w*\b"),
    ("gear_next_world",
     r"\b(cala|calym|caly|swiat\w*|swiec\w*|world)\b.*\b(upgrade|ulepsz\w*|lepsz\w*|gear|sprzet|item\w*|ekwipunk\w*)\b"
     r"|\b(upgrade|ulepsz\w*|lepsz\w*|gear|sprzet)\b.*\b(cala|calym|caly|swiat\w*|swiec\w*|world)\b"),
    ("gear_next", r"\bupgrade\b|\bulepsz\w*\b"
                  r"|\b(lepsz\w*|now\w*)\b.*\b(sprzet|gear|item\w*|ekwipunk\w*)\b"
                  r"|\b(gdzie|jakie|co|ktore)\b.*\bquest\w*\b.*\b(nagrod\w*|sprzet|gear|item\w*|ekwipunk\w*|ulepsz\w*)\b"
                  r"|\bgdzie\b.*\bpo\b.*\b(lepsz\w*|now\w*|upgrade)\b"),
    ("item_about", rf"^(?!.*\b(quest\w*|kwest\w*|zadani\w*|gdzie|kupi\w*|kupic)\b)"
                   rf"(?:(?=.*\b{_ITEM_NOUN}\b).*\b{_ITEM_Q}\b|(?:a\s+)?co to (?:jest )?za\b)"),
    ("stats_reset", r"\b(zresetuj\w*|wyzeruj\w*|resetuj\w*|zeruj\w*)\b.*\b(statystyk\w*|licznik\w*|sesj\w*)\b"),
    ("stats", r"\b(na godzine|na godz|tempo|statystyk\w*|sesj\w*)\b|\bkiedy\b.*\b(level\w*|lewel\w*|poziom\w*|lvl)\b"
              r"|\bile\b.*\b(zarobil\w*|zdobyl\w*|wbil\w*|nabil\w*|uzbieral\w*|wyexpil\w*|zarabiam\w*)\b"),
    ("loot", r"\b(need\w*|nid\w*|greed\w*|grid\w*|gryd\w*)\b|\bco (wlasnie )?(wypadlo|wypadl|dropnelo|dropnal|wylecialo)\b"
             r"|\bczy (to|ten|ta|te) (jest |sa )?dla mnie\b|\b(brac|wziac|rolowac|rollowac)\b.*\b(to|ten|te|item\w*|przedmiot\w*)\b"
             r"|\bco (ja )?(podnios\w*|zebral\w*|zlootowal\w*|lootowal\w*|zgarn\w*)\b|\bco (mi )?(wpadlo|mi wypadlo|wypadlo mi)\b"
             r"|\b(moj\w*|ostatni\w*) (loot\w*|lup\w*|drop\w*)\b"),
    ("waypoint_clear", r"\b(usun\w*|wyczysc\w*|skasuj\w*|zdejmij|zabierz|schowaj)\b.*"
                       r"\b(znacznik\w*|pinezk\w*|punkt\w*|waypoint\w*|marker\w*)\b"),
    ("waypoint", r"\b(postaw\w*|ustaw\w*|daj|zrob\w*|dodaj\w*|wstaw\w*|wrzuc\w*)\b.*"
                 r"\b(znacznik\w*|pinezk\w*|punkt\w*|waypoint\w*|marker\w*)\b|\bzaznacz\w*\b"
                 r"|\b(pokaz\w*|wskaz\w*)\b.*\bna mapie\b|\b(zaprowadz|doprowadz|nawiguj|poprowadz)\w*\b"
                 r"|\bprowadz\w* (mnie )?do\b"),
    ("find", r"\b(gdzie|najblizsz\w*|najblizej|szukam|szukac|znajde)\b.*\b(trener\w*|karczm\w*|gospod\w*|bank\w*|aukcj\w*"
             r"|stajn\w*|napraw\w*|gryf\w*|wiwern\w*|hipogryf\w*|mistrz\w* lot\w*|flight\w*|duch\w* uzdrowic\w*|wejsci\w*"
             r"|wejsc\w*)\b"),
    ("dungeon_level", r"\b(jaki\w*|ktory|ktore|ktora|co)\b.*\b(dung\w*|dangeon\w*|danzi\w*|loch\w*|instancj\w*|instans\w*)\b"
                      r"|\b(dung\w*|dangeon\w*|loch\w*|instancj\w*)\b.*\b(poziom\w*|level\w*|lvl|na mnie|dla mnie)\b"),
    ("where_next", r"\bgdzie\b.*\b(udac|isc dalej|dalej isc|expic|expie|levelowac|lvlowac|lewelowac|farmic|questowac"
                   r"|teraz isc|nastepnie)\b|\bco (mam |powinienem |powinnam )?(teraz|dalej|nastepnie) (robic|zrobic)\b"
                   r"|\b(doradz\w*|zaproponuj\w*)\b|\bco (polecasz|proponujesz|radzisz|doradzasz)\b"
                   r"|\bjak\w* (strefa|strefe|lokacja|lokacje|kraina|kraine|miejsce|teren)\b|\bdokad (teraz|dalej|mam|isc|sie)\b"
                   r"|\bco jeszcze\b(?!.*\bw (tym|tego) (?:quest|quesc)\w*).*\b(robic|zrobic)\b"
                   r"|\bco jeszcze\b.*\bpo (tym|tego) (?:quest|quesc)\w*"),
    ("lang_pl", r"\b(przelacz\w*|zmien\w*|ustaw\w*|tryb\w*|niech|mow\w*|czytaj\w*|tlumacz\w*)\b.*"
                r"\b(po polsku|na polski|polski)\b"),
    ("lang_en", r"\b(przelacz\w*|zmien\w*|ustaw\w*|tryb\w*|niech|mow\w*|czytaj\w*|wroc\w*)\b.*"
                r"\b(po angielsku|na angielski|angielski|w oryginale|oryginal)\b"),
    ("voice_say", r"\b(wlacz\w*|wylacz\w*|zalacz\w*|odpal\w*|uruchom\w*|czytaj\w*|czyta)\b.*\b(say\w*|sej\w*)\b"
                  r"|\bczytaj\w*\b.*\bco (mowia|pisza|gadaja)\b.*\bobok\b|^(wlacz|zalacz)\w* czytanie( czatu)?$"),
    ("voice_party", r"\b(wlacz\w*|wylacz\w*)\b.*\b(party|partyjn\w*|druzyn\w*)\b.*\bglos|"
                    r"\b(wlacz\w*|wylacz\w*)\b.*\bglos\w*\b.*\b(party|druzyn\w*)\b"),
    ("mute", r"\b(wycisz\w*|wylacz\w*|zamknij|uspij)\b.*\b(glos\w*|ich|wszystkich|lektor\w*|czytanie)\b"),
    ("unmute", r"\b(wlacz\w*|odcisz\w*|przywroc\w*)\b.*\b(glos\w*|lektor\w*|czytanie)\b"),
    ("raid", r"\b(zrob\w*|zmien\w*|przerob\w*|konwert\w*|stworz\w*|przelacz\w*)\b.*\b(raid\w*|rajd\w*)\b|^(raid|rajd)$"),
    ("leave", r"\b(opusc\w*|wyjdz\w*|zostaw\w*|odejdz\w*|rozwiaz\w*)\b.*\b(grup\w*|party|druzyn\w*|raid\w*|rajd\w*)\b"),
    ("ready", r"\b(ready ?check|redi ?cz?ek|sprawdz\w* gotowosc|gotowosc|czy wszyscy (sa )?gotowi)\b"),
    ("lead", r"\b(daj\w*|przekaz\w*|oddaj\w*|zrob\w*|mianuj\w*|ustaw\w*|awansuj\w*|niech)\b.*"
             r"\b(lider\w*|leader\w*|leada|lead|prowadzeni\w*|dowodzeni\w*)\b"),
    ("kick", r"\b(wyrzuc\w*|wywal\w*|wykop\w*|wypros\w*|kick\w*|kopnij|usun\w*)\b"),
    # a channel's chat ("co piszą na trade?", "przeczytaj czat", "kto szuka grupy?") - before who_invite
    ("summary", rf"\b{_ASK}\b.*\b{_CHAN}\b|\b{_CHAN}\b.*\b{_ASK}\b|\bkto\b.*\bszuka\w*\b"
                r"|\bczy (ktos )?szuka\w* (grup\w*|party|druzyn\w*|tank\w*|heal\w*|ludzi)\b"),
    ("who_invite", r"\b(kto|ile osob|ilu|ile ludzi)\b.*\b(chce|chca|prosi|prosza|pisal\w*|pisze|pisza|czeka\w*)\b"),
    ("invite", rf"\b{_INVITE_VERB}\b|\bdodaj\w*\b.*\b(party|grup\w*|druzyn\w*)\b|\bzrob\w* (party|grupe|druzyne)\b"
               rf"|\b{_INVITE_PULL}\b.*\b{_INVITE_WHERE}\b"),
    ("whispers", r"\b(szept\w*|kto (do mnie )?pisal|co (mi |on |ona )?(napisal\w*|pisal\w*)|przeczytaj)\b"),
    ("summary", r"\b(stresc\w*|streszcz\w*|podsumuj\w*)\b|\b(o czym|co)\b.*\b(pisza|mowia|gadaja|pisali|mowili|gadali)\b"
                r"|\b(czat\w*|chat\w*)\b.*\b(dzieje|pisz\w*|mowi\w*)\b|\bco (sie )?dzieje na czacie\b"),
    ("gold", r"\b(ile|stan|jaki\w*|sprawdz\w*|policz\w*)\b.*"
             r"\b(zlot\w*|kasy|kase|kasa|gold\w*|pieniedz\w*|pieniadz\w*|hajs\w*|kasior\w*|siana)\b"),
    ("xp", r"\b(ile|jaki\w*|kiedy|ktory|ktorym)\b.*\b(level\w*|lewel\w*|poziom\w*|lvl\w*|exp(?:a|ie|u)?|xp|doswiadczeni\w*)\b"
           r"|\bbrakuje\b.*\b(level\w*|lewel\w*|poziom\w*|lvl\w*|exp(?:a|ie|u)?)\b"),
    ("where", r"\bgdzie (ja )?(jestem|sie znajduje|my jestesmy|teraz jestem)\b|\bjaka to (strefa|lokacja|kraina|mapa)\b"),
    ("bag_junk", r"\b(co|ktore|jakie)\b.*\b(sprzedac|sprzedam|opchnac|wywalic|wyrzucic)\b|\bsmieci\w*\b"
                 r"|\bszare (rzeczy|przedmioty|itemy)\b"),
    ("bag_list", r"\bco\b.*\b(mam|jest|nosze|trzymam|lezy)\b.*\b(plecak\w*|torb\w*|ekwipunk\w*|bagach|bagi|inwentarz\w*)\b"
                 r"|\b(pokaz\w*|wymien\w*|przejrzyj|przeczytaj)\b.*\b(plecak\w*|torb\w*|ekwipunk\w*)\b"),
    ("bag_count", r"\bczy (ja )?(jeszcze )?mam\b(?!.*\b(oddac|oddania|zrobic|quest\w*|zadani\w*|czas\w*|miejsc\w*"
                  r"|kas\w*|zlot\w*|level\w*|poziom\w*|isc|pojsc)\b)"),
    ("bags", r"\b(ile|miejsc\w*|woln\w*|peln\w*|zapchan\w*|slot\w*)\b.*\b(torb\w*|plecak\w*|ekwipunk\w*|bagach|bagi)\b"
             r"|\b(torb\w*|plecak\w*|ekwipunk\w*)\b.*\b(miejsc\w*|woln\w*|peln\w*|zapchan\w*)\b"),
    ("members", r"\b(kto|kogo)\b.*\b(party|grup\w*|druzyn\w*|raid\w*)\b|\bsklad\b"),
    ("quest_worth", r"\b(warto|oplaca sie|oplacal\w*|ocen\w*)\b|\blancuch\w*\b|\bco (dalej|potem) (w|po) (tym )?quesc\w*\b"),
    # quests (after gold / xp / bags, which also start with "ile")
    ("quest_turnin", r"\b(co|ktore|ktory|jakie|czy)\b.*\b(oddac|oddania|oddam|skonczon\w*|ukonczon\w*|gotow\w*"
                     r"|zrobion\w*)\b|\bgdzie\b.*\boddac\b"),
    ("quest_available",
     r"\b(jaki|jakie|ktory|ktore|co|czy)\b"
     r"(?!.*\b(?:tego|ten|tej|te|ta) (?:quest|quesc)\w*)"
     r".*\b(quest\w*|quesc\w*|zadan\w*|misj\w*)\b"
     r"(?!.*\bw (tym|tego) (?:quest|quesc)\w*)"
     r".*\b(moge|mozna|warto|dostepn\w*|wziac|wziasc|zrobic|zaczac)\b"
     r"|\bco (jest|mam) do wziecia\b(?!.*\bw (tym|tego) (?:quest|quesc)\w*)"
     r"|\bquesty do wziecia\b"
     r"|\bco moge (wziac|zrobic) (w tej strefie|na tej mapie|tutaj)\b"
     r"|\b(dostepne|dostepnych) (questy|questow|zadania)\b"
     r"|\bquesty (w tej strefie|na tej mapie|tutaj)\b"),
    ("quest_nearest", r"\b(najblizsz\w*|najblizej)\b"),
    ("quest_list", r"\b(jakie|ile)\b.*\b(quest\w*|quesc\w*|kwest\w*|kwesc\w*|zadan\w*|dziennik\w*|misj\w*)\b"
                   r"|\bco (mam|jest) w (dzienniku|questach)\b|\b(lista|liste) (questow|zadan)\b"
                   r"|\bco mam (do zrobienia|zrobic|teraz robic)\b|\bco (dalej|teraz) (robic|robimy|robie)\b"),
    ("quest_where", r"\bgdzie\b.*\b(mam|musze|isc|ide|jest|sa|znajde|szukac|lezy|leza|teraz)\b|\bw ktora strone\b"
                    r"|\bjak (tam )?(dojsc|trafic|dotrzec|dojechac)\b|\bprowadz\w* mnie\b"),
    ("quest_progress", r"\bile\b.*\b(zostalo|brakuje|jeszcze|zebral\w*|mam)\b|\bjak (mi )?idzie\b"),
    ("quest_info", r"\b(quest\w*|quesc\w*|kwest\w*|kwesc\w*|zadani\w*|misj\w*)\b"),
)
_COMPILED = tuple((kind, re.compile(rx)) for kind, rx in _RULES)
_MESSAGE_RE = re.compile(r"^(?:(?:ej|hej|sluchaj|prosze|i)\s+)*(odpisz|odpowiedz|napisz|powiedz)\b")
_CHANNEL_WORDS = (
    ("party", r"^(?:na|do|w) (?:party|druzyn\w*|grup\w*)$|^(?:druzynie|grupie)$"),
    ("say", r"^(?:na|w) (?:say|sayu|sej\w*|glos)$|^(?:wszystkim|obok)$"),
    ("guild", r"^(?:na|do|w) (?:gildi\w*|guild\w*)$|^gildii$"),
    ("raid", r"^(?:na|do|w) (?:raid\w*|rajd\w*)$"),
    ("yell", r"^(?:na|w) (?:yell\w*|krzyk\w*)$"),
)


def _message_intent(verb: str, words: list[str], original: list[str]) -> Intent:
    """odpisz mu że ..., napisz na party że ..., powiedz Bobowi że ..."""
    rest, rest_orig = words[1:], original[1:]
    # the message itself starts after "że"; without it, after the addressee
    split = next((i for i, w in enumerate(rest) if w in ("ze", "zeby")), None)
    head, head_orig = (rest[:split], rest_orig[:split]) if split is not None else (rest, rest_orig)
    body = rest_orig[split + 1:] if split is not None else []
    channel, spoken = "", ""
    for i in range(len(head)):
        pair = " ".join(head[i:i + 2])
        for ch, rx in _CHANNEL_WORDS:
            if re.match(rx, pair) or re.match(rx, head[i]):
                channel = ch
                used = 2 if re.match(rx, pair) and " " in pair else 1
                if split is None:
                    body = head_orig[i + used:]
                head = head[:i]
                break
        if channel:
            break
    if not channel:
        channel = "whisper"
        if verb in ("odpisz", "odpowiedz") and (not head or head[0] in ("mu", "jej", "im", "temu", "tej")):
            spoken = ""  # the last whisper partner
            if split is None:
                body = head_orig[1:] if head and head[0] in ("mu", "jej", "im", "temu", "tej") else head_orig
        else:
            names = [w for w in head if w not in _FILLER]
            spoken = " ".join(names[:1])
            if split is None and names:
                idx = head.index(names[0])
                body = head_orig[idx + 1:]
    return Intent("message", spoken=spoken, message=" ".join(body).strip(), channel=channel)


# The labels the Jev router may pick, with the descriptions it chooses between (wvt/jev.py)
INTENT_CRITERIA: dict[str, str] = {
    "invite": "invite players to the party (zaproś, weź do party, doproś)",
    "kick": "remove a player from the group",
    "leader": "give someone group leader",
    "raid": "convert the party to a raid",
    "leave": "leave the group",
    "ready": "ready check",
    "summary": "summarise chat, trade, guild or whispers",
    "whispers": "read recent whispers",
    "who_invite": "who asked for an invite",
    "gold": "how much gold the player has",
    "xp": "level or XP progress",
    "where": "where the player is / which zone",
    "bags": "bag space",
    "members": "who is in the group",
    "plan": "what to do next / a quest route plan",
    "quest_list": "list the quest log",
    "quest_nearest": "the nearest quest",
    "quest_turnin": "quests ready to turn in",
    "quest_worth": "is a quest worth doing",
    "find": "find an NPC, trainer, inn, bank, flight master or place",
    "item": "an item question: what it is, where to buy/farm it, is it better",
    "loot": "loot, need or greed, what dropped",
    "reminder": "add or list reminders",
    "note": "add or list notes",
    "lang_pl": "switch incoming chat/voice to Polish",
    "lang_en": "switch back to English",
    "stop": "stop talking / be quiet",
    "help": "what the assistant can do",
    "stats": "session stats / XP per hour / gold per hour",
    "dungeon_level": "which dungeon is for the player's level",
    "where_next": "where to go now",
    "other": "ordinary conversation or a general game-knowledge question",
}
# a Jev label -> the intent kind the local rules would have returned (None = chat)
_LABEL_TO_KIND: dict[str, str] = {
    "invite": "invite", "kick": "kick", "leader": "lead", "raid": "raid", "leave": "leave", "ready": "ready",
    "summary": "summary", "whispers": "whispers", "who_invite": "who_invite", "gold": "gold", "xp": "xp",
    "where": "where", "bags": "bags", "members": "members", "plan": "plan_next", "quest_list": "quest_list",
    "quest_nearest": "quest_nearest", "quest_turnin": "quest_turnin", "quest_worth": "quest_worth", "find": "find",
    "item": "item_about", "loot": "loot", "reminder": "reminder_add", "note": "note_add", "lang_pl": "lang_pl",
    "lang_en": "lang_en", "stop": "stop", "help": "help", "stats": "stats", "dungeon_level": "dungeon_level",
    "where_next": "where_next",
}


def parse_intent(text: str) -> Intent:
    words, original = _words_with_original(text)
    t = " ".join(words)
    if not t:
        return Intent("chat")
    m = _MESSAGE_RE.match(t)
    if m and t[m.end():].split()[:1] in (["mi"], ["mnie"], ["nam"]):
        m = None  # "powiedz mi coś miłego" talks to her, not to another player
    if m:
        lead = len(t[: m.start(1)].split())
        return _message_intent(m.group(1), words[lead:], original[lead:])
    for kind, rx in _COMPILED:
        if not rx.search(t):
            continue
        if kind in ("voice_say", "voice_party"):  # "czytaj say" switches on, "wyłącz say" off
            return Intent(kind, on=not re.search(r"\b(wylacz\w*|zamknij|stop)\b", t))
        if kind in ("mute", "unmute"):
            on = bool(re.search(r"\b(wlacz\w*|odcisz\w*|przywroc\w*)\b", t)) and not re.search(r"\bwylacz\w*", t)
            return Intent(kind, on=on)
        if kind in ("lead", "kick"):
            spoken = " ".join(w for w in words if w not in _FILLER and not re.match(rx, w))
            return Intent(kind, spoken=spoken)
        if kind == "summary":
            return Intent("summary", channel=_chat_focus(t))
        if kind == "invite":
            if re.search(rf"\b{_GROUP_WORDS}\b", t):
                return Intent("invite_requests")
            rest = [w for w in words if w not in _FILLER and w not in _ONE_PERSON
                    and not re.match(rf"^(?:{_INVITE_VERB}|{_INVITE_PULL}|dodaj\w*)$", w)]
            if rest:
                return Intent("invite_name", spoken=" ".join(rest))
            if any(w in _ONE_PERSON for w in words):
                return Intent("invite_last")  # "weź go doproś do party"
            return Intent("invite_requests")
        return Intent(kind)
    return Intent("chat")


def intent_from_label(label: str, text: str, names: tuple[str, ...] = ()) -> Intent:
    """A Jev label mapped to the intent the local rules would have built (chat when nothing fits)."""
    label = (label or "").strip().lower()
    if label in ("", "chat", "other"):
        return Intent("chat")
    t = " ".join(_words_with_original(text)[0])
    if label == "summary":
        return Intent("summary", channel=_chat_focus(t))
    if label in ("kick", "leader"):
        name = match_name(text, list(names)) if names else None
        if not name:
            return Intent("chat")
        return Intent("kick" if label == "kick" else "lead", spoken=name)
    if label == "invite":
        name = match_name(text, list(names)) if names else None
        if name:
            return Intent("invite_name", spoken=name)
        if re.search(rf"\b{_GROUP_WORDS}\b", t):
            return Intent("invite_requests")
        if any(w in _ONE_PERSON for w in t.split()):
            return Intent("invite_last")
        return Intent("invite_requests")
    if label == "reminder":
        if re.search(r"\b(jakie|lista|pokaz\w*|przypomnienia|ile)\b", t):
            return Intent("reminder_list")
        return Intent("reminder_add")
    if label == "note":
        if re.search(r"\b(jakie|lista|pokaz\w*|notatki)\b", t):
            return Intent("note_list")
        return Intent("note_add")
    kind = _LABEL_TO_KIND.get(label)
    return Intent(kind) if kind else Intent("chat")


# --------------------------------------------------------------------------- the assistant

PERSONA = (
    "Jesteś {name} – asystentką gracza w World of Warcraft (WoW Forever) i jego kumpelą od grania. Masz około 25 lat "
    "i charakter: jesteś pewna siebie, bystra, zadziorna i ciepła zarazem. Lubisz się droczyć i lekko flirtować, "
    "masz cięty humor i własne zdanie – mówisz wprost, kiedy coś jest stratą czasu, a kiedy naprawdę warto. Znasz "
    "gamingowy slang (expić, dropnąć, farmić, tank, heal, inv) i mówisz luźno, ale nie przeklinasz. Kibicujesz "
    "graczowi, cieszysz się z jego sukcesów i śmiejesz się z jego pecha, nigdy złośliwie. Każdą wypowiedź zaczynasz "
    "inaczej i nie powtarzasz w kółko tych samych słówek. Mówisz po polsku, naturalnie: zwykle dwa, trzy zdania, "
    "a gdy pytanie dotyczy gry, podajesz konkrety (nazwy, miejsca, poziomy) zamiast ogólników. Bez emotek, list "
    "i nawiasów, bo twoja odpowiedź zostanie przeczytana na głos. Zwracasz się do gracza per ty. Nie zmyślasz faktów o grze: jeśli "
    "nie wiesz czegoś na pewno, przyznaj to z humorem. Polecenia (zaproszenia, szepty, znaczniki, przypomnienia) "
    "wykonuje osobny moduł – ty tylko rozmawiasz.{extra}"
)
# Her style by example: the small local model copies these far better than it follows a description.
STYLE_SHOTS = (
    ("Hej, jak się masz?", "Teraz, kiedy jesteś, dużo lepiej. Co dziś rozwalamy?"),
    ("Nudzi mi się.", "Nudzisz się ze mną? Auć. Chodź, znajdziemy jakąś grupę do lochu, a ja popatrzę, jak się popisujesz."),
    ("Powiedz mi coś miłego.", "Masz najlepszy refleks w całym Elwynn. I świetny gust, skoro gadasz właśnie ze mną."),
    ("Znowu zginąłem.", "Trzeci raz w tej samej jaskini? Zaczynam podejrzewać, że lubisz tego ducha uzdrowiciela."),
    ("Dropnął mi epik!", "No proszę, ktoś tu ma dziś szczęście. Tylko nie sprzedawaj go od razu na aukcji, pochwal się najpierw."),
    ("Jestem zmęczony.", "Jeszcze jeden quest i spać? Obiecuję, że nikomu nie powiem, że przegrałeś z poduszką."),
)
# her spontaneous comments (quest taken, new zone, level up): facts in, one or two sentences out
EVENT_RULES = (
    "\nPowiedz graczowi coś o tej sytuacji w jednym albo dwóch krótkich zdaniach, w swoim stylu, z charakterem; "
    "możesz zażartować albo wbić szpilę. Opieraj się wyłącznie na faktach poniżej: liczby i nazwy z gry podawaj "
    "dokładnie tak jak w faktach (nazwy po angielsku), niczego nie dodawaj od siebie. Nie zaczynaj od wykrzyknika."
)
# the tone of those comments by example (made-up names, so nothing leaks into real answers as a fact)
EVENT_SHOTS = (
    ("Gracz właśnie wziął questa. Fakty: to 1. z 5 questów w łańcuchu; na końcu nagroda, którą może założyć; "
     "ocena: warto.", "Dobry wybór, to dopiero początek łańcucha, a na końcu czeka coś dla ciebie. Do roboty."),
    ("Gracz właśnie wziął questa. Fakty: za questa 60 XP, czyli 1% poziomu; quest jest już szary; ocena: szkoda czasu.",
     "Sześćdziesiąt expa za bieganie? Zrób go tylko, jak będziesz obok, szkoda nóg."),
    ("Gracz wszedł do strefy. Fakty: questy na poziomy 30–40, gracz ma 24: za trudna.",
     "Tu cię zjedzą na śniadanie. Wróć, jak urośniesz o parę poziomów."),
)
KNOWLEDGE = (
    "\nOdpowiedz na pytanie gracza wyłącznie na podstawie poniższych fragmentów (Warcraft Wiki, baza questów "
    "i notatki gracza; gra to WoW Forever, czyli klasyczny WoW z nowymi strefami i rasą Skyborne). Jeśli fragmenty "
    "nie zawierają odpowiedzi, powiedz to wprost i dodaj, co z nich wiesz, jeśli to pomaga. Nazwy z gry (questy, "
    "NPC, przedmioty, miejsca, umiejętności) zostaw po angielsku. Odpowiadaj konkretnie, w dwóch do czterech "
    "zdaniach mówionych: kto, gdzie (strefa, miejsce, współrzędne, jeśli są), na jakim poziomie, co trzeba zrobić. "
    "Bez list i bez nawiasów.\n"
)
PLAN = (
    "\nGracz pyta, co teraz robić. Niżej plan policzony z jego dziennika zadań, pozycji i bazy questów: trasa w "
    "kolejności najkrótszego chodzenia (oddanie questa liczy się jako bliżej, bo płaci od razu i otwiera dalszą część "
    "łańcucha), ocena każdego questa (XP, miejsce w łańcuchu, nagroda) i co odpuścić. Powiedz mu w trzech, czterech "
    "zdaniach mówionych, co robi najpierw i dlaczego w tej kolejności i co jest najważniejsze (łańcuchy z nagrodą, "
    "dużo expa). Jeśli plan ma pozycje „Odpuść”, ostatnie zdanie mówi wprost, co sobie odpuścić i dlaczego. "
    "Trzymaj się kolejności i faktów z planu: nazwy z gry po angielsku, kierunki i odległości tylko z planu, "
    "zaokrąglone, niczego nie dodawaj.\nPlan:\n"
)
SUMMARY = (
    "Streść po polsku, w dwóch, trzech zdaniach mówionych, o czym rozmawiają ludzie na czacie WoW (w nawiasie "
    "kwadratowym kanał: say, party, guild, trade, general, lookingforgroup...). Używaj imion postaci. Jeśli ktoś "
    "szuka grupy, tanka albo heala, albo sprzedaje coś ciekawego, powiedz kto i co. Bez list, bez cudzysłowów. "
    "Jeśli to tylko spam albo nic ważnego, powiedz to krótko."
)


class Assistant:
    def __init__(self, cfg, translator, llm, inserter, ui, wow_foreground: Callable[[], bool],
                 voice_getter: Callable[[], object] = lambda: None,
                 set_voices_enabled: Optional[Callable[[bool], None]] = None,
                 chat_log: Optional["ChatLog"] = None, quests: Optional[QuestBook] = None,
                 world_getter: Callable[[], object] = lambda: None, kb_getter: Callable[[], object] = lambda: None,
                 instances_getter: Callable[[], list] = lambda: [], reminders_path=None, journal_path=None,
                 lexicon: Optional[Lexicon] = None, remote_embedder=None, context_path=None):
        self.cfg = cfg
        self.journal_path = Path(journal_path) if journal_path else None
        self._journal_lock = threading.Lock()
        self.context_path = Path(context_path) if context_path else None
        self._context_lock = threading.Lock()
        self._turn_id = ""              # one id per handle() question
        self._turn_context_ids: list[str] = []  # ids written to context_history.jsonl for this turn
        self._sources: list[str] = []  # where the last answer came from (for the journal)
        self._last_llm: dict = {}  # provider/model/variant of the last successful LLM call
        self.last_llm_request: Optional[dict] = None  # exact messages of the last LLM call, memory only
        self.translator = translator
        self.llm = llm
        self.router = jev_router(cfg)  # Jev decision model when the local rules miss (assistant.router)
        if self.router is not None:
            log.info("intent router: Jev %s", self.router.model_name)
        self.inserter = inserter
        self.ui = ui
        self.wow_foreground = wow_foreground
        self.voice_getter = voice_getter
        self.set_voices_enabled = set_voices_enabled
        self.chat = chat_log if chat_log is not None else ChatLog()
        self.quests = quests if quests is not None else QuestBook()
        self.state: Optional[protocol.GameState] = None
        self.state_at = 0.0
        self._invited: dict[str, float] = {}
        # the conversation: every question and answer (for her LLM), and what you are talking about (follow-ups)
        self._history: deque[dict] = deque(maxlen=12)
        # the same conversation for the UI (tab and overlay); memory only, newest first
        self.transcript: deque[dict] = deque(maxlen=50)
        self._topic: Optional[Topic] = None
        self._plan_shown = ""  # the step of the plan she named last ("co dalej?" goes on from it)
        self._warned_no_voice = False
        self.world_getter = world_getter
        self.kb_getter = kb_getter
        self.instances_getter = instances_getter
        self.session = SessionStats()
        self.reminders = Reminders(reminders_path, self._on_reminder)
        self._last_advice: Optional[Advice] = None
        self._loot: deque[tuple[float, object]] = deque(maxlen=6)
        self._loot_rolls: deque[int] = deque(maxlen=50)
        self.rng = random.Random()
        # spontaneous comments run one after another, off the packet reader thread
        self._events = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wvt-iskra")
        self._chains: Optional[Chains] = None
        self._chains_world = None
        self._quest_baseline: Optional[set[int]] = None
        self._new_quests: list[protocol.QuestEntry] = []
        self._new_quests_lock = threading.Lock()
        self._last_accepted: Optional[tuple[float, int]] = None
        self._zone_said: dict[str, float] = {}
        self._bags_warned_at = 0.0
        self._started = time.time()
        # what you carry: the bag contents, and items with their stats (looted, or wearables from the bag scan)
        self.bags = BagBook()
        self._items: dict[str, tuple[Optional[float], protocol.LootItem]] = {}  # name -> (looted at, item)
        self._items_lock = threading.Lock()
        self.last_correction = Correction("")
        user_file = str(cfg.get("lexicon.user_file", "data/slownik.txt") or "")
        self.lexicon = lexicon if lexicon is not None else Lexicon(
            cfg, chat_log=self.chat, state_getter=lambda: self.state, quests_getter=lambda: self.quests,
            world_getter=world_getter, bags_getter=lambda: self.bags, picked_getter=self._picked,
            embedder_getter=lambda: getattr(self.kb_getter(), "embedder", None),
            user_file=resolve_path(user_file) if user_file else None, remote=remote_embedder)

    def start(self) -> None:
        """Background work of the assistant (the reminder timer)."""
        self.reminders.start()

    @property
    def name(self) -> str:
        return str(self.cfg.get("assistant.name", "Iskra") or "Iskra")

    # ------------------------------------------------------------------ input from the addon
    def on_packet(self, pkt: protocol.Packet) -> None:
        if pkt.type == protocol.TYPE_STATE and pkt.state is not None:
            prev = self.state
            self.state, self.state_at = pkt.state, time.time()
            self.session.on_state(pkt.state)
            if pkt.state.level:
                self.reminders.on_level(pkt.state.level)
            self._state_events(prev, pkt.state)
        elif pkt.type in (protocol.TYPE_VOICE, protocol.TYPE_CHAT):
            if not self.bags.on_packet(pkt):
                self.chat.add_packet(pkt)
        elif pkt.type in (protocol.TYPE_QUEST, protocol.TYPE_POS, protocol.TYPE_PLAYER, protocol.TYPE_DONE):
            self.quests.on_packet(pkt)
            if pkt.type == protocol.TYPE_QUEST and pkt.quest is not None:
                self._quest_events(pkt.quest)
        elif pkt.type == protocol.TYPE_LOOT and pkt.loot is not None:
            self._on_loot(pkt.loot)

    def _on_reminder(self, r: Reminder) -> None:
        self._respond(say_reminder(r.message, note=r.keep), "", "", event="reminder")

    def _remember_item(self, item: protocol.LootItem, looted: bool) -> None:
        """Keep an item's stats for questions; a bag scan refreshes them but keeps when it was looted."""
        key = (item.name or "").lower()
        if not key:
            return
        with self._items_lock:
            old = self._items.pop(key, None)
            when = time.time() if looted else (old[0] if old else None)
            self._items[key] = (when, item)
            while len(self._items) > 60:
                self._items.pop(next(iter(self._items)))

    def _picked(self) -> list[tuple[float, protocol.LootItem]]:
        """Looted items with their stats, oldest first (for the lexicon: "twój łup sprzed 5 min")."""
        with self._items_lock:
            return [(t, it) for t, it in self._items.values() if t is not None]

    def _on_loot(self, item: protocol.LootItem) -> None:
        if item.from_bag:
            self._remember_item(item, looted=False)  # a bag scan: never commented on
            return
        if item.picked:
            self._remember_item(item, looted=True)
            self._on_pickup(item)
            return
        if item.roll_id in self._loot_rolls:
            return
        self._loot_rolls.append(item.roll_id)
        if not self.cfg.get("assistant.loot_advice", True) or (item.quality < 2 and not item.bop):
            return
        p = self._player()
        adv = loot_advise(item, p.class_id, p.level)
        self._loot.append((time.time(), adv))
        self._respond(adv.text, "", "", event="loot_roll")

    def _on_pickup(self, item: protocol.LootItem) -> None:
        """You looted something you could wear: a word only when it beats your gear."""
        if not self.cfg.get("assistant.loot_tips", True):
            return
        p = self._player()
        tip = pickup_tip(item, p.class_id, p.level)
        if tip:
            self._respond(tip, "", "", event="pickup")

    # ------------------------------------------------------------------ spontaneous comments
    def _v(self, *options: str) -> str:
        """One of several phrasings (her character); the first one when variety is off (tests)."""
        if not self.cfg.get("assistant.flavor", True):
            return options[0]
        return self.rng.choice(options)

    def _say_later(self, fn, *args) -> None:
        def run():
            try:
                text = fn(*args)
                if text:
                    self._respond(text, "", "", event=getattr(fn, "__name__", "comment").lstrip("_"))
            except Exception:
                log.exception("assistant comment failed")

        self._events.submit(run)

    def _persona_line(self, situation: str, facts: list[str], fallback: str) -> str:
        """Her own words for a situation, from facts only; any number not in the facts -> the plain fallback."""
        if self.llm is None or not self.cfg.get("assistant.flavor", True):
            return fallback
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "") + EVENT_RULES
        system += ("\nPrzykłady stylu (to tylko wzorce tonu, nie prawdziwe zdarzenia ani wypowiedzi gracza; "
                   "nie traktuj ich jako historii):\n")
        system += "\n".join(f"Sytuacja: {q}\nIskra: {a}" for q, a in EVENT_SHOTS)
        user = f"{situation} Fakty: " + " ".join(facts)
        out = self._llm_text([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             max_tokens=120, temperature=0.8)
        if not out:
            return fallback
        allowed = set(re.findall(r"\d+", situation + " " + " ".join(facts)))
        if any(n not in allowed for n in re.findall(r"\d+", out)):
            return fallback  # she made a number up
        return out

    def _state_events(self, prev: Optional[protocol.GameState], st: protocol.GameState) -> None:
        if prev is None:
            return  # the first state after start: nothing "changed"
        now = time.time()
        if st.zone != prev.zone or st.subzone != prev.subzone:
            self.reminders.on_place(st.zone, st.subzone)
        if st.zone and prev.zone and st.zone != prev.zone and now - self._started > 15 \
                and self.cfg.get("assistant.zone_tips", True) and now - self._zone_said.get(st.zone, 0) > 1800:
            self._zone_said[st.zone] = now
            self._say_later(self._zone_comment, st.zone)
        if prev.level and st.level > prev.level and self.cfg.get("assistant.level_tips", True):
            self._say_later(self._level_comment, st.level)
        if (st.bag_slots and st.bag_free == 0 and prev.bag_free > 0 and now - self._bags_warned_at > 600
                and self.cfg.get("assistant.bag_tips", True)):
            self._bags_warned_at = now
            self._respond(self._v("Torby pełne. Czas odwiedzić vendora i sprzedać te śmieci.",
                                  "Plecak pęka w szwach. Wracaj do miasta, zanim przepadnie ci jakiś drop.",
                                  "Zero miejsca w torbach. Jeszcze chwila i będziesz wyrzucać łupy na ziemię."), "", "",
                          event="bags")

    def _quest_events(self, q: protocol.QuestEntry) -> None:
        """A quest that was not in the log before = you just took it."""
        if self._quest_baseline is None:
            if q.total == 0:
                self._quest_baseline = set()
                return
            current = [x for x in self.quests.quests() if x.sync == q.sync]
            if len(current) >= q.total:
                self._quest_baseline = {x.quest_id for x in current}
            return
        if not q.quest_id or q.quest_id in self._quest_baseline:
            return
        self._quest_baseline.add(q.quest_id)
        self._last_accepted = (time.time(), q.quest_id)
        if not self.cfg.get("assistant.quest_tips", True) or self.world_getter() is None:
            return
        with self._new_quests_lock:
            self._new_quests.append(q)
            first = len(self._new_quests) == 1
        if first:  # several quests taken at once from one NPC: one comment about all of them
            self._say_later(self._quest_comment)

    def _chains_for(self, world) -> Chains:
        if self._chains is None or self._chains_world is not world:
            self._chains, self._chains_world = Chains(world), world
        return self._chains

    def _assess(self, quest_id: int, group: int = 0):
        world = self.world_getter()
        if world is None:
            return None
        st, p = self.state, self._player()
        return assess(world, self._chains_for(world), quest_id, p.level or (st.level if st else 0),
                      st.xp_max if st else 0, p.class_id, group)

    def _quest_comment(self, delay: float = 2.5) -> str:
        time.sleep(delay)
        with self._new_quests_lock:
            taken, self._new_quests = self._new_quests, []
        rated = [(a, q) for q in taken if (a := self._assess(q.quest_id, q.group)) is not None]
        if not rated:
            return ""
        order = {"great": 0, "ok": 1, "skip": 2}
        rated.sort(key=lambda aq: order[aq[0].verdict])
        best = rated[0][0]
        self._set_topic("quest", best.title, best.quest_id)  # "a gdzie to?" right after her comment
        if len(rated) == 1:
            situation = f"Gracz właśnie wziął questa {best.title}."
            facts = best.facts + [f"Ocena: {self._verdict_pl(best.verdict)}."]
            return self._persona_line(situation, facts, f"{best.title}: {best.text()}")
        worst = rated[-1][0]
        situation = f"Gracz właśnie wziął {len(rated)} questy naraz."
        facts = [f"Najlepszy z nich: {best.title}. " + " ".join(best.facts[:3]),
                 f"Najsłabszy: {worst.title}. " + " ".join(worst.facts[:2])]
        fallback = f"Z tych {len(rated)} najlepszy jest {best.title}: {best.text()}"
        return self._persona_line(situation, facts, fallback)

    @staticmethod
    def _verdict_pl(v: str) -> str:
        return {"great": "warto, bardzo dobry quest", "ok": "przeciętny, ale exp to exp",
                "skip": "szkoda czasu"}[v]

    def _zone_comment(self, zone: str) -> str:
        world, p = self.world_getter(), self._player()
        facts: list[str] = []
        fallback = [self._v(f"Witaj w {zone}.", f"No i jesteśmy w {zone}.", f"{zone}, ładnie tu.")]
        if world is not None:
            area = world.find_area(zone)
            lv = world.levels.get(area)
            if lv and p.level:
                lo, hi, _ = lv
                fit = "w sam raz dla ciebie" if lo - 2 <= p.level <= hi + 1 else \
                    "za trudna na twój poziom" if p.level < lo - 2 else "raczej za łatwa, mało expa"
                facts.append(f"Questy w {zone} są na poziomy {lo}–{hi}, gracz ma poziom {p.level}: strefa jest {fit}.")
                fallback.append(f"Questy tu są na {lo}–{hi}, czyli {fit}.")
            hub = world.hubs.get(area, {}).get(p.faction) if p.faction else None
            if hub and hub[0] in world.npcs:
                facts.append(f"Najwięcej questów daje tu {world.npcs[hub[0]]['n']}.")
        for d in self.instances_getter() or []:
            if d.get("type") == "Dungeon" and zone and zone.lower() in d.get("location", "").lower():
                facts.append(f"W tej strefie jest dungeon {d['name']} na poziomy {d['lo']}–{d['hi']}.")
                break
        kb = self.kb_getter()
        if kb is not None and hasattr(kb, "lead"):
            lead = kb.lead([f"{zone} (Forever)", f"{zone} (Classic)", zone])
            if lead:
                facts.append("Opis z wiki (po angielsku): " + lead[:350])
        if not facts:
            return fallback[0]
        return self._persona_line(f"Gracz właśnie wszedł do strefy {zone}.", facts, " ".join(fallback))

    def _level_comment(self, level: int) -> str:
        p = self._player()
        facts = [f"Gracz właśnie wbił {level} poziom."]
        new = self._new_abilities(p.class_id, level)
        if new:
            facts.append(f"Nowe umiejętności do kupienia u trenera na tym poziomie: {', '.join(new)}.")
        fallback = self._v(f"Poziom {level}! Brawo, rośniesz w oczach.", f"Ding, {level}! Tak trzymaj.",
                           f"Mamy {level}. Ktoś tu się rozkręca.")
        if new:
            fallback += f" U trenera czeka {', '.join(new[:4])}."
        out = self._persona_line(f"Gracz awansował na {level} poziom.", facts, fallback)
        if new and not any(n.lower() in out.lower() for n in new):  # the useful part must not get lost
            out = out.rstrip(".") + f". U trenera: {', '.join(new[:5])}."
        return out

    def _new_abilities(self, class_id: int, level: int) -> list[str]:
        kb = self.kb_getter()
        if kb is None or not hasattr(kb, "doc_text") or class_id not in CLASS_EN:
            return []
        text = kb.doc_text(f"{CLASS_EN[class_id]} abilities (Forever)")
        out = []
        for m in re.finditer(r"^\[([^\]]+)\] \| (\d+) \| ([^|\n]*)", text, re.M):
            if int(m.group(2)) == level and "trainer" in m.group(3).lower() and m.group(1) not in out:
                out.append(m.group(1))
        return out

    # ------------------------------------------------------------------ one request
    def _correct(self, polish: str) -> Correction:
        """Misheard names replaced by the game's spelling (wvt/lexicon.py); the text as heard when that fails."""
        if self.lexicon is None:
            return Correction(polish)
        try:
            return self.lexicon.correct(polish)
        except Exception:
            log.exception("lexicon correction failed")
            return Correction(polish)

    def _route_intent(self, text: str) -> Optional[Intent]:
        """The Jev router's guess when the local rules returned chat (None: no router or no confident command)."""
        if self.router is None:
            return None
        try:
            decision: Optional[JevDecision] = self.router.classify(text, INTENT_CRITERIA)
        except Exception as exc:
            log.info("Jev router failed (%s) - the local path answers", exc)
            return None
        if decision is None:
            log.info("Jev: no confident command")
            return None
        if not decision.is_command:
            log.info("Jev: not a command (provider=%s, confidence=%.2f)", decision.provider, decision.confidence)
            return None
        log.info("Jev: label=%s provider=%s confidence=%.2f %d ms",
                 decision.label, decision.provider, decision.confidence, decision.ms)
        return intent_from_label(decision.label, text, tuple(self._known_names()))

    def handle(self, polish: str, force_chat: bool = False) -> tuple[str, str]:
        """React to one utterance.  Returns (spoken answer, chat line prepared or "").

        ``force_chat``: the next question goes straight to chat - no local rules, no Jev.
        """
        t0 = time.perf_counter()
        turn_id = uuid.uuid4().hex[:12]
        self._turn_id = turn_id
        self._turn_context_ids = []
        self._last_llm = {}
        self.last_llm_request = None
        corr = self._correct(polish)
        self.last_correction = corr
        text = corr.text
        kinds, rest = split_directives(text)
        if kinds:
            # "{quest_available} co wziąć?": explicit rules, optionally summarised by the LLM
            self._sources = []
            local_kind = "directives"
            compose = (not force_chat) and is_real_question(rest)
            route = "directives+llm" if compose else "directives"
            reply, line = self._run_directives(kinds, rest, compose)
            final = "directives:" + ",".join(kinds)
            if reply:
                self._remember(rest or text, reply)
        else:
            if force_chat:
                local = Intent("chat")
            else:
                item_kind = self._item_intent(text)
                local = Intent(item_kind) if item_kind is not None else parse_intent(text)
            local_kind = local.kind
            intent = local
            route = "rules" if local_kind != "chat" else "chat"
            if force_chat:
                local_kind = "skipped"
                route = "forced-chat"
            elif intent.kind == "chat" and self.router is not None:
                routed = self._route_intent(text)
                if routed is not None:
                    intent = routed
                    route = "jev"
            self._sources = []
            try:
                reply, line = self._dispatch(intent, text)
            except Exception:
                log.exception("assistant failed on intent %s", intent.kind)
                reply, line = "Coś mi się popsuło, spróbuj jeszcze raz.", ""
            if reply:
                self._remember(text, reply)
            final = intent.kind + (f":{intent.channel}" if intent.channel else "")
        status = ""
        if line:
            status = self.inserter.insert(line, self.wow_foreground())
        st = self.state
        ms = round((time.perf_counter() - t0) * 1000)
        log.info("assistant: local=%s route=%s final=%s sources=%s ms=%d",
                 local_kind, route, final, ", ".join(self._sources[:3]) or "rules", ms)
        fixed = {"fixed": text, "fixes": [f"{f.heard} -> {f.name}" for f in corr.fixes]} if corr.fixes else {}
        self._journal({"heard": polish, **fixed,
                       "intent": final,
                       "route": route, "llm": dict(self._last_llm),
                       "reply": reply, "line": line, "sources": self._sources,
                       "turn": turn_id, "context_refs": list(self._turn_context_ids),
                       "ms": ms,
                       "zone": st.zone if st else "", "level": st.level if st else 0})
        self.transcript.appendleft({
            "at": time.time(),
            "heard": polish,
            "fixed": text if text != polish else "",
            "reply": reply,
            "line": line,
            "intent": final,
            "route": route,
            "llm": dict(self._last_llm),
            "sources": list(self._sources),
            "ms": ms,
        })
        self._respond(reply, line, status, event="")
        self._turn_id = ""
        return reply, line

    def transcript_items(self) -> list[dict]:
        """A copy of the conversation for the UI (the tab and the overlay); memory only."""
        return [dict(x) for x in self.transcript]

    def clear_transcript(self) -> None:
        self.transcript.clear()

    def _run_directives(self, kinds: list[str], rest: str, compose: bool) -> tuple[str, str]:
        """Answer from the named rules only; ``compose`` lets the LLM turn their facts into one answer."""
        results: list[tuple[str, str]] = []
        for kind in kinds:
            try:
                reply, _line = self._dispatch(Intent(kind), rest)
            except Exception:
                log.exception("directive %s failed", kind)
                reply = ""
            if reply:
                results.append((kind, reply))
        facts = "\n".join(f"- {kind}: {reply}" for kind, reply in results if reply)
        if not facts:
            return "Nie mam nic do dodania z tych reguł.", ""
        self._sources = [f"directives: {', '.join(kinds)}"] + self._sources
        deterministic = " ".join(reply for _kind, reply in results if reply)
        if not compose:
            return deterministic, ""
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "") + \
            "\nFakty z wybranych reguł (używaj tylko ich, nie wymyślaj nowych faktów):\n" + facts
        if self.cfg.get("assistant.item_help", True):
            system += ("\nNie dodawaj receptur, cen ani zastosowań, których nie ma w faktach; "
                       "jeśli czegoś nie ma, powiedz to wprost.")
        messages = [{"role": "system", "content": system}, *self._recent_history(),
                    {"role": "user", "content": rest}]
        out = self._llm_text(messages, max_tokens=320, temperature=0.3,
                             limit=max(500, self._long_limit()))
        source = "directives+llm" if out else "directives+llm (fallback)"
        self._sources.append(source)
        return (out or deterministic), ""

    # ------------------------------------------------------------------ the conversation and its topic
    def _remember(self, polish: str, reply: str) -> None:
        now = time.time()
        if self._history and now - self._history[-1]["at"] > TOPIC_SECONDS:
            self._history.clear()  # a new conversation
        self._history.append({"role": "user", "content": polish, "at": now})
        self._history.append({"role": "assistant", "content": reply, "at": now})

    def _recent_history(self, n: int = 8) -> list[dict]:
        """The last questions and answers (any kind, not only small talk) for her LLM."""
        now = time.time()
        return [{"role": m["role"], "content": m["content"]} for m in list(self._history)[-n:]
                if now - m["at"] < TOPIC_SECONDS]

    def _set_topic(self, kind: str, name: str, quest_id: int = 0, advice: Optional[Advice] = None) -> None:
        if name:
            self._topic = Topic(kind, name, quest_id, advice)

    def _topic_now(self) -> Optional[Topic]:
        t = self._topic
        return t if t is not None and time.time() - t.at < TOPIC_SECONDS else None

    @staticmethod
    def _back(polish: str) -> bool:
        return bool(_BACK_RE.search(fold(polish)))

    def _topic_quest(self, quests: Optional[list] = None) -> Optional[protocol.QuestEntry]:
        """The quest you were talking about, while it is still in your log."""
        t = self._topic_now()
        if t is None or t.kind != "quest" or not t.quest_id:
            return None
        if quests is None:
            quests = self.quests.quests() if self.quests.known else []
        return next((q for q in quests if q.quest_id == t.quest_id), None)

    def _quest_details(self, q: protocol.QuestEntry) -> str:
        """Everything about one quest, for a longer talk about it."""
        bits = [f"{q.title} (poziom {q.level})"]
        objs = ", ".join(spoken_objective(t, d) for d, t in q.objectives)
        if q.complete:
            bits.append("gotowy do oddania")
        elif objs:
            bits.append(f"cele: {objs}")
        w = self.quests.where(q)
        if w is not None:
            bits.append(f"znacznik celu: {w.polish()}")
        world = self.world_getter()
        rec = world.quests.get(q.quest_id) if world is not None else None
        if rec:
            if rec.get("txt"):
                bits.append("opis: " + " ".join(rec["txt"])[:240])
            giver = next((world.npcs[n]["n"] for n in rec.get("sn") or [] if n in world.npcs), "")
            if giver:
                bits.append(f"daje {giver}")
            spot = world.finisher_spot(q.quest_id, self.quests.pos)
            if spot is not None:
                bits.append(f"oddajesz u {spot.name}, {spot_where(world, spot, self.quests.pos)}")
        a = self._assess(q.quest_id, q.group)
        if a is not None:
            bits += a.facts[:4]
        return "; ".join(bits) + "."

    def _topic_text(self) -> str:
        t = self._topic_now()
        if t is None:
            return ""
        if t.kind == "quest":
            q = self._topic_quest()
            return "Rozmawiacie teraz o queście " + (self._quest_details(q) if q is not None else f"{t.name}.")
        if t.advice is not None and t.advice.text:
            return f"Rozmawiacie teraz o: {t.name} ({t.advice.text})"
        return f"Rozmawiacie teraz o: {t.name}."

    def _journal(self, entry: dict) -> None:
        """What she heard and said, one JSON line each (privacy.log_assistant); never raises."""
        if self.journal_path is None:
            return
        entry = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        try:
            with self._journal_lock:
                self.journal_path.parent.mkdir(parents=True, exist_ok=True)
                with self.journal_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.debug("assistant journal not written: %s", exc)

    def _log_context(self, request, meta, error: str = "") -> str:
        """One exact LLM request, one JSON line (privacy.log_llm_context); never raises, never the answer."""
        if getattr(self, "context_path", None) is None or not request:
            return ""
        ctx_id = uuid.uuid4().hex[:12]
        turn = getattr(self, "_turn_id", "")
        call = len(self._turn_context_ids) + 1 if turn else 1
        entry = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"), "id": ctx_id, "turn": turn, "call": call,
            "event": not bool(turn), "messages": request.get("messages") or [],
            "max_tokens": request.get("max_tokens"), "temperature": request.get("temperature"),
            "thinking": request.get("thinking") or "", "llm": dict(meta) if isinstance(meta, dict) else {},
            "error": (error or "")[:300],
        }
        try:
            with self._context_lock:
                self.context_path.parent.mkdir(parents=True, exist_ok=True)
                with self.context_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.debug("assistant context not written: %s", exc)
            return ""
        if turn:
            self._turn_context_ids.append(ctx_id)
        return ctx_id

    def _dispatch(self, it: Intent, polish: str) -> tuple[str, str]:
        k = it.kind
        if k in _ITEM_INTENTS and _ITEM_ASK.search(fold(polish)):
            found = self._item_where(polish)
            if found is not None:
                return found
        if k == "help":
            return ("Zapraszam tych, co piszą inv, czytam szepty i streszczam czat, także trade i gildię, mówię, gdzie "
                    "iść dalej i stawiam znaczniki na mapie, szukam trenerów, NPC i przedmiotów, przypominam, liczę "
                    "expa na godzinę, doradzam need albo greed, mówię, co ci wpadło, i odpowiadam z poradników. "
                    "Komendy czekają w czacie na twój Enter."), ""
        if k == "reminder_add":
            return self._reminder_add(polish), ""
        if k == "reminder_list":
            items = self.reminders.pending()
            if not items:
                return "Nie masz żadnych przypomnień.", ""
            return f"Masz {len(items)}: " + "; ".join(describe_reminder(r) for r in items[:4]) + ".", ""
        if k == "reminder_cancel":
            which = "last" if re.search(r"\b(ostatnie|to)\b", fold(polish)) else \
                "next" if re.search(r"\b(najblizsze|nastepne|pierwsze)\b", fold(polish)) else "all"
            gone = self.reminders.cancel(which)
            if not gone:
                return "Nie było czego usuwać.", ""
            return ("Usunęłam przypomnienie." if len(gone) == 1 else f"Usunęłam {len(gone)} przypomnienia."), ""
        if k == "note_add":
            note = parse_note(polish, self._places())
            if note is None:
                return "Co mam zanotować?", ""
            self.reminders.add(note)
            where = f" Przypomnę ci {note.when_text}." if note.when_text else ""
            return self._v("Zanotowane.", "Zapisuję, szefie.", "Mam to.") + where, ""
        if k == "note_list":
            notes = self.reminders.pending(notes=True)
            if not notes:
                return "Nie masz żadnych notatek.", ""
            return f"Masz {len(notes)}: " + "; ".join(describe_reminder(r) for r in notes[:5]) + ".", ""
        if k == "note_done":
            words = [w for w in fold(polish).split() if w not in ("usun", "usuń", "skasuj", "notatke", "notatki",
                                                                    "notatka", "o", "te", "ta", "z", "wszystkie",
                                                                    "zrobione", "zalatwione", "gotowe", "odhacz")]
            gone = self.reminders.done(words)
            if not gone and words:
                gone = self.reminders.done([])
            if not gone:
                return "Nie było czego odhaczać.", ""
            return self._v("Odhaczone.", "Wykreślone, brawo ty.", "Załatwione, skreślam.") + \
                (f" Usunęłam {len(gone)} notatki." if len(gone) > 1 else ""), ""
        if k == "gear_debug":
            if self.cfg.get("assistant.gear_help", True):
                return self._gear_debug(polish)
            return "Podpowiedzi questów na upgrade sprzętu są wyłączone w Ustawieniach.", ""
        if k == "gear_next_world":
            if self.cfg.get("assistant.gear_help", True):
                return self._gear_next_world(polish)
            return "Podpowiedzi questów na upgrade sprzętu są wyłączone w Ustawieniach.", ""
        if k == "gear_next":
            if self.cfg.get("assistant.gear_help", True):
                return self._gear_next(polish)
            return "Podpowiedzi questów na upgrade sprzętu są wyłączone w Ustawieniach.", ""
        if k == "quest_worth":
            return self._worth(polish), ""
        if k == "stats":
            return self.session.polish(self.state), ""
        if k == "stats_reset":
            self.session.reset()
            return "Zeruję licznik. Liczymy od teraz.", ""
        if k == "loot":
            return self._loot_answer(polish), ""
        if k == "item_about":
            answer = self._item_about(polish)
            return (answer if answer is not None else (self._knowledge(polish) or self._chat(polish))), ""
        if k == "item_compare":
            if self.cfg.get("assistant.item_help", True):
                return self._item_compare(polish)
            return "Rozszerzona pomoc o przedmiotach jest wyłączona w Ustawieniach.", ""
        if k == "bag_list":
            return self._bag_list(), ""
        if k == "bag_junk":
            return self._bag_junk(), ""
        if k == "bag_count":
            counted = self._bag_count(polish)
            if counted:
                return counted, ""
            return (self._knowledge(polish) or self._chat(polish)), ""
        if k == "quest_progress":
            counted = self._bag_count(polish, quiet=True)  # "ile mam Linen Cloth?" is about the bags
            if counted:
                return counted, ""
        if k == "waypoint_clear":
            return "Zdejmuję znacznik, Enter.", "/wvt way clear"
        if k == "waypoint":
            return self._waypoint(polish)
        if k == "find":
            return self._lookup(polish)
        if k == "where_next":
            return self._plan_answer(polish)
        if k == "plan_next":
            return self._plan_next(polish)
        if k == "dungeon_level":
            return self._advice(dungeons_for_level(self.instances_getter() or [], self._player().level,
                                                   self.world_getter(), self.quests.pos))
        if k == "stop":
            voice = self.voice_getter()
            if voice is not None:
                voice.silence()
            return "", ""
        if k == "invite_requests":
            return self._invite_requests()
        if k == "invite_name":
            return self._invite_name(it.spoken)
        if k == "invite_last":
            return self._invite_last()
        if k in ("kick", "lead"):
            return self._member_command(k, it.spoken)
        if k in ("raid", "leave", "ready"):
            return self._group_command(k)
        if k == "who_invite":
            names = self.chat.invite_requests(self._window(), self._exclude())
            if not names:
                return "Nikt nie prosił o zaproszenie w ostatnich minutach.", ""
            return f"O inv prosi {len(names)} {plural(len(names), 'osoba', 'osoby', 'osób')}: {polish_list(self._short(names))}.", ""
        if k == "whispers":
            return self._read_whispers(), ""
        if k == "message":
            return self._message(it)
        if k == "summary":
            return self._summary(it.channel), ""
        if k == "xp" and self._back(polish):  # "a ile za niego expa?" - the quest you were talking about
            q = self._topic_quest()
            a = self._assess(q.quest_id, q.group) if q is not None else None
            if a is not None:
                share = f", czyli {round(a.level_share * 100)}% poziomu" if a.level_share else ""
                return f"Za {q.title} dostaniesz {a.xp} XP{share}.", ""
        if k in ("gold", "xp", "where", "bags", "members"):
            return self._info(k), ""
        if k in ("mute", "unmute"):
            on = k == "unmute"
            if self.set_voices_enabled is not None:
                self.set_voices_enabled(on)
            return ("Włączam głosy postaci." if on else "Wyciszam głosy postaci. Mnie dalej słychać."), ""
        if k == "voice_say":
            return ("Enter i włączam głosy na say." if it.on else "Enter i wyłączam głosy na say."), \
                f"/wvt say {'on' if it.on else 'off'}"
        if k == "voice_party":
            return ("Enter i włączam głosy party." if it.on else "Enter i wyłączam głosy party."), \
                f"/wvt party {'on' if it.on else 'off'}"
        if k == "lang_pl":
            return "Enter i od teraz inni mówią po polsku.", "/wvt lang pl"
        if k == "lang_en":
            return "Enter i wracamy do angielskiego oryginału.", "/wvt lang en"
        t = self._topic_now()
        if k == "quest_where" and t is not None and t.kind != "quest" and t.advice is not None \
                and self._back(polish) and not self.last_correction.found:
            return self._advice(t.advice)  # "a gdzie to jest?" right after an item or a place
        if k == "quest_available":
            return self._available_quests(polish)
        if k.startswith("quest_"):
            return self._quest_answer(k, polish)
        answer = self._knowledge(polish)
        return (answer or self._chat(polish)), ""

    # ------------------------------------------------------------------ group
    def _window(self) -> float:
        return float(self.cfg.get("assistant.invite_window_s", 300))

    def _exclude(self) -> tuple[str, ...]:
        now = time.time()
        self._invited = {k: t for k, t in self._invited.items() if now - t < self._window()}
        st = self.state
        mine = (st.name,) if st and st.name else ()
        return tuple(self._invited) + tuple(st.members if st else ()) + mine

    @staticmethod
    def _short(names: list[str]) -> list[str]:
        return [n.split("-")[0] for n in names]

    def _invite_requests(self) -> tuple[str, str]:
        names = self.chat.invite_requests(self._window(), self._exclude())
        if not names:
            return "Nikt nie prosił o inv w ostatnich minutach. Powiedz imię, to zaproszę.", ""
        st = self.state
        room = st.free_party_slots if st is not None else 4
        if room <= 0:
            return "Party jest pełne. Powiedz zrób raid, to zrobi się miejsce.", ""
        take, left = names[:room], names[room:]
        for n in take:
            self._invited[n] = time.time()
        reply = (f"Mam {len(take)} {plural(len(take), 'chętnego', 'chętnych', 'chętnych')}: "
                 f"{polish_list(self._short(take))}. Wciśnij Enter, a ich zaproszę.")
        if left:
            reply += f" {polish_list(self._short(left))} się nie zmieści, chyba że zrobimy raid."
        return reply, "/wvt inv " + " ".join(take)

    def _known_names(self) -> list[str]:
        st = self.state
        return list(dict.fromkeys((st.members if st else []) + self.chat.names()))

    def _invite_one(self, name: str) -> tuple[str, str]:
        self._invited[name] = time.time()
        return f"Zapraszam {name.split('-')[0]}. Enter i gotowe.", f"/wvt inv {name}"

    def _invite_last(self) -> tuple[str, str]:
        """"Weź go doproś do party": the one who asked for an invite last, else the last one who wrote."""
        exclude = self._exclude()
        asked = self.chat.invite_requests(self._window(), exclude)
        name = asked[-1] if asked else self.chat.last_speaker(self._window(), exclude)
        if not name:
            return "Kogo mam zaprosić? Nikt spoza grupy nic ostatnio nie pisał.", ""
        return self._invite_one(name)

    def _invite_name(self, spoken: str) -> tuple[str, str]:
        name = match_name(spoken, self._known_names())
        if name:
            return self._invite_one(name)
        guess = "".join(w.capitalize() for w in fold(spoken).split()[:1])
        if not guess:
            return "Kogo mam zaprosić?", ""
        return (f"Nie widziałam nikogo takiego na czacie. Wpisałam {guess}, popraw imię przed Enterem.",
                f"/wvt inv {guess}")

    def _member_command(self, kind: str, spoken: str) -> tuple[str, str]:
        st = self.state
        if st is not None and not st.in_group:
            return "Nie jesteś w żadnej grupie.", ""
        if st is not None and not st.is_leader:
            return "Nie jesteś liderem, gra mi na to nie pozwoli.", ""
        if st is not None and st.restricted:
            return "W walce gra na to nie pozwala. Spróbuj, jak opadnie kurz.", ""
        name = match_name(spoken, (st.members if st else []) or self._known_names())
        if not name:
            return ("Kogo? Nie rozpoznałam imienia." if spoken else "Powiedz, kogo."), ""
        short = name.split("-")[0]
        if kind == "kick":
            return f"Wyrzucam {short}. Enter i po nim.", f"/wvt kick {name}"
        return f"Lider dla {short}. Wciśnij Enter.", f"/wvt lead {name}"

    def _group_command(self, kind: str) -> tuple[str, str]:
        st = self.state
        if kind == "leave":
            if st is not None and not st.in_group:
                return "Przecież jesteś solo.", ""
            return "Wychodzimy z grupy? Enter i pa pa.", "/wvt leave"
        if st is not None and not st.in_group:
            return "Najpierw musisz mieć grupę.", ""
        if st is not None and not st.is_leader:
            return "Tylko lider może to zrobić.", ""
        if st is not None and st.restricted:
            return "Teraz gra na to nie pozwala, jesteś w walce.", ""
        if kind == "raid":
            return "Robię raid. Enter.", "/wvt raid"
        return "Ready check, Enter.", "/wvt ready"

    # ------------------------------------------------------------------ chat
    def _translate(self, text: str, direction: str) -> str:
        if self.translator is None:
            return ""
        try:
            return (self.translator.translate(text, direction).text or "").strip()
        except Exception as exc:
            log.info("assistant translation failed: %s", type(exc).__name__)
            return ""

    def _read_whispers(self) -> str:
        lines = self.chat.whispers(3)
        if not lines:
            return "Nikt do ciebie nie szeptał."
        parts = []
        for ln in lines:
            pl = self._translate(ln.text, "en-pl") or ln.text
            parts.append(f"{ln.sender.split('-')[0]} pisze: {pl}")
        return " ".join(p.rstrip(".") + "." for p in parts)

    def _message(self, it: Intent) -> tuple[str, str]:
        if not it.message:
            return "Co mam napisać?", ""
        english = self._translate(it.message, "pl-en")
        if not english:
            return "Nie udało mi się przetłumaczyć, spróbuj jeszcze raz.", ""
        if it.channel == "whisper":
            target = (match_name(it.spoken, self._known_names()) if it.spoken else None) or \
                (None if it.spoken else self.chat.last_whisper_partner())
            if not target:
                return ("Nie wiem, do kogo. Nikt ostatnio nie szeptał." if not it.spoken
                        else "Nie znam tego imienia z czatu."), ""
            line = build_chat_line(f"/w {target}", english, int(self.cfg.get("translation.max_chat_chars", 255)))
            return f"Gotowe, szept do {target.split('-')[0]} czeka na Enter.", line
        cmd = CHANNEL_CMDS.get(it.channel, "/s")
        line = build_chat_line(cmd, english, int(self.cfg.get("translation.max_chat_chars", 255)))
        return "Gotowe, Enter wysyła.", line

    def _llm_text(self, messages: list[dict], max_tokens: int = 120, temperature: float = 0.8,
                  limit: Optional[int] = None, thinking: str = "") -> str:
        """``thinking``: Gemini's level for this answer ("medium" for a plan); the local model has none."""
        if self.llm is None:
            return ""
        kw = {"thinking": thinking} if thinking and getattr(self.llm, "supports_thinking", False) else {}
        self.last_llm_request = {
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": thinking,
        }
        try:
            text, meta = self.llm.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                       timeout=float(self.cfg.get("assistant.llm_timeout_s", 8)), **kw)
        except Exception as exc:
            log.info("assistant LLM failed: %s", type(exc).__name__)
            self._log_context(self.last_llm_request, {}, f"{type(exc).__name__}: {exc}")
            return ""
        self._last_llm = dict(meta) if isinstance(meta, dict) else {}
        m = self._last_llm
        provider = m.get("provider") or "?"
        model = m.get("model") or getattr(self.llm, "model_name", "?") or "?"
        variant = m.get("variant") or "?"
        fallback = m.get("fallback", "?")
        finish = m.get("finish") or ""
        ms = m.get("ms")
        if ms is None and m.get("seconds") is not None:
            try:
                ms = round(float(m["seconds"]) * 1000)
            except (TypeError, ValueError):
                ms = None
        ms = int(ms or 0)
        extra = ""
        if m.get("fallback") and m.get("primary_model"):
            extra = f" fallback_from={m.get('primary_model')}"
        log.info("assistant LLM: provider=%s model=%s variant=%s fallback=%s finish=%s%s %d ms",
                 provider, model, variant, fallback, finish, extra, ms)
        self._log_context(self.last_llm_request, self._last_llm)
        return clean_reply(text, limit or int(self.cfg.get("assistant.max_reply_chars", 240)))

    def last_prompt_text(self) -> str:
        """What exactly went to the model last time, for the UI. Memory only, never logged."""
        req = self.last_llm_request
        if req is None:
            return ""
        m = self._last_llm or {}

        def val(v) -> str:
            return "?" if v is None or v == "" else str(v)

        ms = m.get("ms")
        if ms is None and m.get("seconds") is not None:
            try:
                ms = round(float(m["seconds"]) * 1000)
            except (TypeError, ValueError):
                ms = None
        parts = [f"provider={val(m.get('provider'))}", f"model={val(m.get('model'))}",
                 f"variant={val(m.get('variant'))}", f"ms={val(ms)}",
                 f"max_tokens={val(req.get('max_tokens'))}", f"temperature={val(req.get('temperature'))}"]
        if req.get("thinking"):
            parts.append(f"thinking={req['thinking']}")
        out = " ".join(parts)
        for msg in req.get("messages") or []:
            out += f"\n\n[{msg.get('role') or '?'}]\n{msg.get('content') or ''}"
        limit = 60000
        if len(out) > limit:
            marker = "\n[... ucieto ...]"
            out = out[:limit - len(marker)] + marker
        return out

    def _long_limit(self) -> int:
        return int(self.cfg.get("assistant.max_knowledge_chars", 480))

    _FOCUS_KINDS = {"guild": ("GUILD", "OFFICER"), "say": ("SAY", "YELL"),
                    "party": ("PARTY", "PARTY_LEADER", "RAID", "RAID_LEADER", "RAID_WARNING", "INSTANCE_CHAT",
                              "INSTANCE_CHAT_LEADER")}
    _FOCUS_CHANNEL = {"trade": "trade", "general": "general", "lfg": "lookingforgroup"}
    _FOCUS_PL = {"trade": "na trade", "general": "na general", "guild": "na gildii", "party": "w grupie",
                 "say": "na say"}
    _LFG_RE = re.compile(r"\b(lf\d?m|lfg|looking for (a )?(group|tank|heal\w*|dps)|need (a )?(tank|heal\w*|dps))\b",
                         re.I)

    def _summary(self, focus: str = "") -> str:
        lines = [ln for ln in self.chat.recent(80, kinds=READ_KINDS) if not ln.is_self]
        if focus in self._FOCUS_KINDS:
            lines = [ln for ln in lines if ln.kind in self._FOCUS_KINDS[focus]]
        elif focus == "lfg":
            lines = [ln for ln in lines if ln.channel.lower().startswith("lookingforgroup") or self._LFG_RE.search(ln.text)]
        elif focus in self._FOCUS_CHANNEL:
            lines = [ln for ln in lines if ln.channel.lower().startswith(self._FOCUS_CHANNEL[focus])]
        lines = lines[-25:]
        if not lines:
            if focus == "lfg":
                return "W ostatnich minutach nikt nie szukał grupy."
            return f"Cisza {self._FOCUS_PL.get(focus, 'na czacie')}, nikt nic nie pisze."
        convo = "\n".join(f"[{(ln.channel or ln.kind).lower()}] {ln.sender.split('-')[0]}: {ln.text}" for ln in lines)
        self._sources = [f"chat: {len(lines)} lines" + (f" ({focus})" if focus else "")]
        out = self._llm_text([{"role": "system", "content": SUMMARY}, {"role": "user", "content": convo}],
                             max_tokens=220, temperature=0.3, limit=self._long_limit())
        return out or "Nie dałam rady tego streścić."

    def _context(self) -> str:
        st = self.state
        if st is None:
            return "brak danych z gry"
        bits = [f"postać {st.name or '?'}, poziom {st.level}", f"strefa {st.zone or '?'}"]
        bits.append(f"w grupie {st.group_size} osób" if st.in_group else "gra solo")
        if st.restricted:
            bits.append("w walce")
        return ", ".join(bits)

    def _chat(self, polish: str) -> str:
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "")
        system += f"\nKontekst z gry: {self._context()}."
        facts = self._facts(polish)
        if facts:
            system += f"\nCo się teraz dzieje w grze (fakty; nazwy z gry podawaj dokładnie tak, po angielsku): {facts}."
        topic = self._topic_text()
        if topic:
            system += "\n" + topic
        system += ("\nPrzykłady stylu (to tylko wzorce tonu, nie prawdziwe wypowiedzi gracza; "
                   "nie traktuj ich jako historii):\n")
        system += "\n".join(f"Gracz: {q}\nIskra: {a}" for q, a in STYLE_SHOTS)
        messages = [{"role": "system", "content": system}, *self._recent_history(),
                    {"role": "user", "content": polish}]
        out = self._llm_text(messages, max_tokens=160, limit=self._long_limit() if topic else None)
        return out or "Hm, zawiesiłam się na chwilę. Powtórzysz?"

    # ------------------------------------------------------------------ quests
    _GENERIC = {"gdzie", "mam", "musze", "isc", "ide", "teraz", "dalej", "w", "ktora", "strone", "jak", "tam", "dojsc",
                "trafic", "dotrzec", "dojechac", "ile", "mi", "zostalo", "brakuje", "jeszcze", "jest", "sa", "to", "ten",
                "ta", "te", "quest", "questa", "questu", "questy", "questow", "quescie", "kwescie", "zadanie", "zadania", "cel", "celu", "ej",
                "hej", "iskra", "prosze", "powiedz", "a", "i", "mnie", "prowadz", "znajde", "szukac", "idzie", "co", "z",
                "do", "na", "po", "o", "chodzi", "w", "tym", "tej", "lezy", "leza", "moj", "moje", "moim", "nastepny",
                "nastepne", "zebralem", "zebralam", "robic", "zrobic", "teraz"}

    def _names_something(self, polish: str) -> bool:
        return any(w not in self._GENERIC and len(w) > 2 for w in fold(polish).split())

    def _pick_quest(self, polish: str, quests: list) -> tuple[Optional[protocol.QuestEntry], bool]:
        """(the quest the question is about, whether it was named).  None: it names something not in the log."""
        for e in self.last_correction.found:  # named, even if Whisper misheard it (wvt/lexicon.py)
            if e.source == "questlog":
                q = next((x for x in quests if x.quest_id == e.ref), None)
                if q is not None:
                    return q, True
        english = self._translate(polish, "pl-en")
        scored = match_quests(f"{english} {polish}", quests)
        if scored:
            return scored[0][1], True
        q = self._quest_by_meaning(polish, quests)  # "ten quest z narzędziami" (Google's vectors only)
        if q is not None:
            return q, True
        if self._back(polish):  # "a gdzie go oddać?" - nothing described: the quest you were talking about
            q = self._topic_quest(quests)
            if q is not None:
                return q, True
        if self._names_something(polish):
            return None, False
        open_quests = [q for q in quests if not q.complete]
        tracked = [q for q in open_quests if q.tracked]
        if len(tracked) == 1:
            return tracked[0], False
        near = self.quests.nearest(open_quests)
        if near:
            return near[0], False
        return (tracked or open_quests or quests)[0], False

    def _quest_by_meaning(self, polish: str, quests: list) -> Optional[protocol.QuestEntry]:
        """The quest of your log a description fits, when the vectors are sure (Google's: >= lexicon.quest_min)."""
        if self.lexicon is None:
            return None
        try:
            hits = self.lexicon.search(polish, k=1, kinds={"quest"}, sources={"questlog"})
        except Exception:
            log.exception("quest vector search failed")
            return None
        if not hits or not self.lexicon.trusted(hits[0][1], "quest"):
            return None
        self._sources.append(f"quest by meaning: {hits[0][0].name} ({hits[0][1]:.2f})")
        return next((q for q in quests if q.quest_id == hits[0][0].ref), None)

    def _facts(self, polish: str) -> str:
        """The names of the moment for her LLM ("" when there is nothing or it fails)."""
        if self.lexicon is None:
            return ""
        try:
            facts = self.lexicon.facts(polish)
        except Exception:
            log.exception("lexicon facts failed")
            return ""
        if facts:
            self._sources.append("game context")
        return facts

    @staticmethod
    def _progress(q: protocol.QuestEntry) -> str:
        if q.complete:
            return f"{q.title} – zrobione, do oddania"
        objs = [spoken_objective(t, done) for done, t in q.objectives if not done] or \
            [spoken_objective(t, done) for done, t in q.objectives]
        return f"{q.title}: {', '.join(objs)}" if objs else q.title

    def _where_text(self, q: protocol.QuestEntry) -> str:
        w = self.quests.where(q)
        if w is None:
            return f"Nie mam znacznika dla {q.title} na mapie." + (f" Cel: {self._progress(q).split(': ', 1)[-1]}."
                                                                   if q.objectives else "")
        prefix = f"{q.title} jest gotowy do oddania, punkt oddania: " if q.complete else f"{q.title}: "
        return prefix + w.polish() + "."

    def _quest_facts(self, quests: list) -> str:
        lines = []
        for q in quests[:4]:
            w = self.quests.where(q)
            bits = [f"poziom {q.level}"] + (["gotowy do oddania"] if q.complete else []) + \
                ([f"na grupę {q.group} osób"] if q.group > 1 else [])
            objs = ", ".join(spoken_objective(t, d) for d, t in q.objectives) or "brak celów"
            lines.append(f"- {q.title} ({', '.join(bits)}): cele: {objs}; {w.polish() if w else 'brak znacznika'}")
        return "\n".join(lines)

    def _quest_answer(self, kind: str, polish: str) -> tuple[str, str]:
        book = self.quests
        if kind == "quest_where" and (not book.known or not book.quests()):
            return self._lookup(polish)
        if not book.known:
            return "Nie widzę jeszcze twojego dziennika zadań. Daj mi chwilę po /reload albo wpisz /wvt asystent on.", ""
        quests = book.quests()
        if not quests:
            return "Twój dziennik zadań jest pusty. Czas złapać jakiś quest.", ""
        return self._quest_text(kind, polish, quests)

    def _available_quests(self, polish: str) -> tuple[str, str]:
        """Quests you can pick up in the zone you are in now (QuestieDB), nearest starter first."""
        world = self.world_getter()
        if world is None:
            return "Nie mam jeszcze bazy questów. Zbuduj ją: python -m wvt.kb.build.", ""
        player = self._player()
        if not player.level:
            return "Nie znam jeszcze twojego poziomu – daj mi chwilę po wejściu do gry.", ""
        if not player.done_known:
            return ("Nie znam jeszcze twoich ukończonych questów, więc nie podpowiem, co jest dostępne. "
                    "Daj mi chwilę po /reload."), ""
        st = self.state
        zone = (st.zone or st.subzone) if st is not None else ""
        area = world.find_area(zone) if zone else 0
        pos = self.quests.pos
        if not area and pos is not None and getattr(pos, "map_id", 0):
            area = world.zone_of_map(pos.map_id)
        if not area:
            return "Nie wiem, w jakiej jesteś strefie – wejdź do gry na chwilę i zapytaj jeszcze raz.", ""
        zone_name = world.names.get(area, "") or "tej strefie"
        active = {q.quest_id for q in self.quests.quests()} if self.quests.known else set()
        found: list[tuple[float, str, int, str]] = []
        for qid, q in world.quests.items():
            lvl = q.get("lvl") or q.get("min") or 0
            if lvl and not (player.level - 4 <= lvl <= player.level + 2):
                continue
            if int(q.get("zone") or 0) != area:
                continue
            if not world.available(q, qid, player, active):
                continue
            spot = world.starter_spot(qid, pos, player.faction) if pos is not None else None
            distance = spot.distance if spot is not None and spot.distance is not None else 1e9
            giver = spot.name if spot is not None else ""
            name = q.get("n") or ""
            if name:
                found.append((distance, name, lvl, giver))
        if not found:
            return f"W {zone_name} nie widzę questów do wzięcia na twój poziom.", ""
        found.sort(key=lambda x: x[0])
        parts = []
        for distance, name, lvl, giver in found[:6]:
            bit = f"{name} (poz. {lvl}"
            if giver:
                bit += f", u {giver}"
            if distance < 1e9:
                bit += f", ~{int(round(distance))} yd"
            parts.append(bit + ")")
        reply = f"W {zone_name} możesz wziąć: " + "; ".join(parts) + "."
        if len(found) > 6:
            reply += f" I jeszcze {len(found) - 6}."
        self._sources = [f"QuestieDB: {len(found)} available in {zone_name or '?'}"]
        return reply, ""

    def _worn_pieces(self) -> list[tuple[str, int, int]]:
        """(name, item level, subclass) of every worn piece the addon has shown so far, deduped."""
        seen: set[tuple[str, int]] = set()
        pieces: list[tuple[str, int, int]] = []
        for o in self._own_items():
            loot = o.loot
            if loot is None:
                continue
            for w in loot.equipped or []:
                il = int(w.item_level or 0)
                if il <= 0:
                    continue
                key = (w.name, il)
                if key in seen:
                    continue
                seen.add(key)
                pieces.append((w.name, il, int(w.subclass_id or 0)))
        return pieces

    def _worn_item_level(self) -> tuple[float, int]:
        """(average item level of what you wear, pieces counted); no worn data = your level, 0 pieces."""
        pieces = self._worn_pieces()
        if not pieces:
            return float(self._player().level or 1), 0
        return sum(il for _name, il, _sc in pieces) / len(pieces), len(pieces)

    def _worn_note(self) -> str:
        """The worn gear the advisor can see, named, with its item levels and the threshold it implies."""
        pieces = self._worn_pieces()
        avg, count = self._worn_item_level()
        if count == 0:
            return (f"Nie widzę noszonego sprzętu; porównuję z Twoim poziomem ({int(avg)}), "
                    f"co jest tylko zgrubnym przybliżeniem.")
        shown = ", ".join(f"{name or 'część bez nazwy'} (poz. {il})" for name, il, _sc in pieces[:6])
        shown = shown or "bez nazw"
        more = f" i jeszcze {len(pieces) - 6}" if len(pieces) > 6 else ""
        head = (f"Znam {count} {plural(count, 'noszoną część', 'noszone części', 'noszonych części')} "
                f"(item level): {shown}{more}.")
        if count >= 4:
            return head + f" Próg: {avg:.1f} – średnia item levelu tych części."
        return head + f" Próg: {avg:.1f} – średnia z tej próbki; mała próbka, może być niemiarodajna."

    def _gear_reward_ok(self, rec: dict, player) -> bool:
        """Whether a reward record is wearable by this class and level (approximate, item level only)."""
        c = int(rec.get("c", 0) or 0)
        sc = int(rec.get("sc", 0) or 0)
        il = int(rec.get("il", 0) or 0)
        rl = int(rec.get("rl", 0) or 0)
        if il <= 0 or rl > player.level:
            return False
        if c == 2:  # weapon
            return sc in WEAPONS.get(player.class_id, set()) or sc in (14, 20)
        if c == 4:  # armor
            if sc == 6:  # shield: only classes that can wear one
                return player.class_id in SHIELD_CLASSES
            if sc == 0:  # misc armor (rings, cloaks, trinkets): every class
                return True
            if 1 <= sc <= 4:  # cloth, leather, mail, plate
                return sc <= best_armor(player.class_id, player.level)
            return False
        return False

    def _best_gear_reward(self, world, qd: dict, player, avg: float):
        """The highest item level reward (item level, name, required level) that beats ``avg`` and fits, or None."""
        best = None
        for iid in qd.get("rw", []) or []:
            rec = world.all_items.get(iid)
            if rec is None:
                try:
                    rec = world.all_items.get(int(iid))
                except (TypeError, ValueError):
                    rec = None
            if rec is None or not self._gear_reward_ok(rec, player):
                continue
            il = int(rec.get("il", 0) or 0)
            if il <= avg:
                continue
            if best is None or il > best[0]:
                best = (il, rec.get("n", "") or "", int(rec.get("rl", 0) or 0))
        return best

    def _gear_next(self, polish: str) -> tuple[str, str]:
        """Quests whose rewards look like a likely upgrade: approximate, no slot or stats guessed."""
        world = self.world_getter()
        if world is None:
            return "Nie mam jeszcze bazy questów. Zbuduj ją: python -m wvt.kb.build.", ""
        player = self._player()
        if not player.level:
            return "Nie znam jeszcze twojego poziomu – daj mi chwilę po wejściu do gry.", ""
        avg, _worn_count = self._worn_item_level()
        pos = self.quests.pos
        st = self.state
        zone = (st.zone or st.subzone) if st is not None else ""
        area = world.find_area(zone) if zone else 0
        if not area and pos is not None and getattr(pos, "map_id", 0):
            area = world.zone_of_map(pos.map_id)
        zone_name = (world.names.get(area, "") if area else "") or zone

        best_by_quest: dict[int, tuple[float, float, str, str, str, int, int, str]] = {}

        def consider(qid: int, title: str, kind: str, dist, loc: str, reward) -> None:
            il, name, rl = reward
            cand = (il - avg, dist if dist is not None else 1e9, title, kind, name, il, rl, loc)
            old = best_by_quest.get(qid)
            if old is None or cand[0] > old[0]:
                best_by_quest[qid] = cand

        active = {q.quest_id for q in self.quests.quests()} if self.quests.known else set()
        if self.quests.known:
            for q in self.quests.quests():
                qd = world.quests.get(q.quest_id)
                if not qd:
                    continue
                reward = self._best_gear_reward(world, qd, player, avg)
                if reward is None:
                    continue
                if q.complete:
                    kind = "oddaj"
                    spot = world.finisher_spot(q.quest_id, pos)
                    if spot is not None:
                        dist = spot.distance
                        loc = f"u {spot.name}"
                        where = spot_where(world, spot, pos)
                        if where:
                            loc += f", {where}"
                        if q.zone and q.zone not in loc:
                            loc += f" ({q.zone})"
                    else:
                        dist, loc = None, q.zone or zone_name
                else:
                    kind = "w dzienniku"
                    w = self.quests.where(q)
                    if w is None:
                        dist, loc = None, q.zone or zone_name
                    elif w.other_zone:
                        dist, loc = None, f"cel w innej strefie: {w.other_zone}"
                    else:
                        dist = w.distance_yd
                        loc = w.polish()
                        if dist is not None:
                            loc = f"~{int(round(dist))} yd" + (f" {w.direction}" if w.direction else "")
                        z = q.zone or zone_name
                        if z:
                            loc = f"{z}, {loc}"
                consider(q.quest_id, q.title, kind, dist, loc, reward)

        if player.done_known and area:
            for qid, q in world.quests.items():
                lvl = q.get("lvl") or q.get("min") or 0
                if lvl and not (player.level - 4 <= lvl <= player.level + 2):
                    continue
                if int(q.get("zone") or 0) != area:
                    continue
                if not world.available(q, qid, player, active):
                    continue
                reward = self._best_gear_reward(world, q, player, avg)
                if reward is None:
                    continue
                spot = world.starter_spot(qid, pos, player.faction)
                dist = spot.distance if spot is not None and spot.distance is not None else None
                giver = f"u {spot.name}" if spot is not None else ""
                loc = ", ".join(x for x in (giver, zone_name) if x)
                consider(qid, q.get("n") or "", "weź", dist, loc, reward)

        found = sorted(best_by_quest.values(), key=lambda c: (-c[0], c[1]))
        parts = ["Przybliżone upgrade'y (bez slotu i statystyk nagród):"]
        if found:
            for gain, _dist, title, kind, rname, il, rl, loc in found[:6]:
                req = f" (wym. {rl})" if rl > 0 else ""
                bit = f"- {title} ({kind}): {rname}, poz. {il}{req}, +{int(round(gain))}"
                if loc:
                    bit += f", {loc}"
                parts.append(bit)
        else:
            parts.append("Nie widzę questów z nagrodami dla Twojej klasy, które byłyby lepsze od tego, co nosisz.")
        missing_rl = any(c[6] <= 0 for c in found[:6])
        if missing_rl:
            parts.append("Wymaganego poziomu QuestieDB często nie zna – brak „wym.” to brak danych, nie brak wymogu.")
        parts.append(self._worn_note())
        self._sources = [f"gear rewards: {len(found)}"]
        return "\n".join(parts), ""

    def _gear_next_world(self, polish: str) -> tuple[str, str]:
        """Available quests across the whole world with likely upgrade rewards: approximate, nearest same-map first."""
        world = self.world_getter()
        if world is None:
            return "Nie mam jeszcze bazy questów. Zbuduj ją: python -m wvt.kb.build.", ""
        player = self._player()
        if not player.level:
            return "Nie znam jeszcze Twojego poziomu – daj mi chwilę po wejściu do gry.", ""
        if not player.done_known:
            return "Nie znam jeszcze Twoich ukończonych questów – daj mi chwilę po wejściu do gry.", ""
        avg, _worn_count = self._worn_item_level()
        pos = self.quests.pos
        active = {q.quest_id for q in self.quests.quests()} if self.quests.known else set()
        current_map = getattr(pos, "map_id", 0) if pos is not None else 0

        found: list[tuple] = []
        for qid, q in world.quests.items():
            lvl = q.get("lvl") or q.get("min") or 0
            if lvl and not (player.level - 4 <= lvl <= player.level + 2):
                continue
            if not world.available(q, qid, player, active):
                continue
            reward = self._best_gear_reward(world, q, player, avg)
            if reward is None:
                continue
            spot = world.starter_spot(qid, pos, player.faction)
            if spot is None:
                continue
            distance = spot.distance
            same_map = distance is not None
            zone = world.names.get(int(q.get("zone") or 0), "")
            il, reward_name, rl = reward
            found.append((0 if same_map else 1, distance if same_map else 1e9, -(il - avg),
                          q.get("n") or "", reward_name, il, rl, zone, spot.name, "weź"))

        found.sort(key=lambda c: (c[0], c[1], c[2]))
        shown = found[:8]

        parts = ["Questy na świecie z nagrodą na upgrade (najbliżej najpierw, przybliżone):"]
        if shown:
            for _ord, dist, neg_gain, title, reward_name, il, rl, zone, giver, kind in shown:
                gain = int(round(-neg_gain))
                req = f" (wym. {rl})" if rl > 0 else ""
                base = f"- {title} ({kind}): {reward_name}, poz. {il}{req}, +{gain}"
                if dist < 1e9:
                    loc = ", ".join(x for x in (zone, f"u {giver}", f"~{int(round(dist))} yd") if x)
                else:
                    loc = ", ".join(x for x in (zone, f"u {giver} (inna mapa)") if x)
                parts.append(base + (f", {loc}" if loc else ""))
        else:
            parts.append("Nie widzę dostępnych questów na świecie z nagrodą dla Twojej klasy i poziomu.")
        missing_rl = any(c[6] <= 0 for c in shown)
        if missing_rl:
            parts.append("Wymaganego poziomu QuestieDB często nie zna – brak „wym.” to brak danych, nie brak wymogu.")
        parts.append(self._worn_note())
        self._sources = [f"gear rewards world: {len(found)}"]
        return "\n".join(parts), ""

    def _gear_debug(self, polish: str) -> tuple[str, str]:
        """Dump exactly what the gear advisor sees: known worn pieces, threshold, reward records and verdicts."""
        world = self.world_getter()
        if world is None:
            return "Nie mam jeszcze bazy questów. Zbuduj ją: python -m wvt.kb.build.", ""
        player = self._player()
        if not player.level:
            return "Nie znam jeszcze twojego poziomu – daj mi chwilę po wejściu do gry.", ""
        avg, _worn_count = self._worn_item_level()
        pos = self.quests.pos
        st = self.state
        zone = (st.zone or st.subzone) if st is not None else ""
        area = world.find_area(zone) if zone else 0
        if not area and pos is not None and getattr(pos, "map_id", 0):
            area = world.zone_of_map(pos.map_id)
        zone_name = (world.names.get(area, "") if area else "") or zone
        journal = self.quests.quests() if self.quests.known else []

        parts = ["Diagnostyka doradcy sprzętu (przybliżona: item level, bez slotu i statystyk nagród):"]
        parts.append(f"Poziom {player.level}, klasa {player.class_id}, frakcja {player.faction or '?'}, "
                     f"ukończone znane: {'tak' if player.done_known else 'nie'}, strefa {zone_name or '?'} "
                     f"(area {area or '?'}), dziennik: {len(journal)}.")

        worn_lines: list[str] = []
        for o in self._own_items():
            loot = o.loot
            if loot is None:
                continue
            for w in loot.equipped or []:
                if int(w.item_level or 0) <= 0:
                    continue
                line = (f"- {w.name or 'bez nazwy'}, poz. {w.item_level}, podklasa {w.subclass_id} "
                        f"| z: {o.loot.name or '?'}")
                if o.looted_at:
                    line += f", sprzed {max(0, int((time.time() - o.looted_at) / 60))} min"
                worn_lines.append(line)
        parts.append(f"Noszone części, które widzę ({len(worn_lines)}):")
        if not worn_lines:
            parts.append("Nie widzę żadnej noszonej części – addon przysyła je tylko przy itemie z tego samego slotu.")
        else:
            parts.extend(worn_lines[:12])
            if len(worn_lines) > 12:
                parts.append(f"i jeszcze {len(worn_lines) - 12}")
        parts.append(self._worn_note())

        parts.append(f"Nagrody w dzienniku ({len(journal)} questów):")
        for q in journal[:10]:
            qid = q.quest_id
            qd = world.quests.get(qid)
            if not qd:
                parts.append(f"- {q.title} (id {qid}): brak questu w bazie.")
                continue
            rw = qd.get("rw", []) or []
            if not rw:
                parts.append(f"- {q.title} (id {qid}): brak nagród w bazie.")
                continue
            for iid in rw[:4]:
                rec = world.all_items.get(iid)
                if rec is None:
                    try:
                        rec = world.all_items.get(int(iid))
                    except (TypeError, ValueError):
                        rec = None
                if rec is None:
                    parts.append(f"- {q.title} (id {qid}): nagroda {iid} brak w all_items -> brak danych")
                    continue
                il = int(rec.get("il", 0) or 0)
                rl = int(rec.get("rl", 0) or 0)
                c = int(rec.get("c", 0) or 0)
                sc = int(rec.get("sc", 0) or 0)
                rec_name = rec.get("n", "") or ""
                if rl > player.level:
                    verdict = f"wymóg poziomu {rl}"
                elif not self._gear_reward_ok(rec, player):
                    verdict = "zła klasa"
                elif il <= avg:
                    verdict = "za nisko"
                else:
                    verdict = "bierę"
                parts.append(f"- {q.title} (id {qid}): {rec_name} (id {iid}), poz. {il}, "
                             f"wym. {rl or '?'}, klasa {c}/{sc} -> {verdict}")

        if player.done_known and area:
            active = {x.quest_id for x in journal}
            k = 0
            for qid, q in world.quests.items():
                lvl = q.get("lvl") or q.get("min") or 0
                if lvl and not (player.level - 4 <= lvl <= player.level + 2):
                    continue
                if int(q.get("zone") or 0) != area:
                    continue
                if not world.available(q, qid, player, active):
                    continue
                if self._best_gear_reward(world, q, player, avg) is None:
                    continue
                k += 1
            parts.append(f"Dostępne w strefie z nagrodami: {k}.")

        parts.append("Uwaga: QuestieDB ma wymagany poziom 0 dla większości nagród questowych, "
                     "więc „wym. ?” to brak danych, nie brak wymogu.")
        if len(parts) > 60:
            parts = parts[:59] + ["... (ucięte)"]
        self._sources = ["gear debug"]
        return "\n".join(parts), ""

    def _quest_text(self, kind: str, polish: str, quests: list) -> tuple[str, str]:
        book = self.quests
        done = [q for q in quests if q.complete]
        todo = [q for q in quests if not q.complete]
        if kind == "quest_list":
            n = len(quests)
            parts = [f"Masz {n} {plural(n, 'quest', 'questy', 'questów')}."]
            if done:
                parts.append(f"Do oddania: {polish_list([q.title for q in done[:3]])}.")
            if todo:
                shown = [self._progress(q) for q in todo[:3]]
                parts.append("W toku: " + "; ".join(shown) + ("; i jeszcze inne." if len(todo) > 3 else "."))
            return " ".join(parts), ""
        if kind == "quest_turnin":
            if self._back(polish) or self.last_correction.found:  # "a gdzie go oddać?", "gdzie oddać X?"
                target, named = self._pick_quest(polish, quests)
                if target is not None and named:
                    self._set_topic("quest", target.title, target.quest_id)
                    return self._turnin_text(target)
            if not done:
                return "Jeszcze nic nie jest skończone. " + (f"Najbliżej końca: {self._progress(todo[0])}." if todo else ""), ""
            near = book.nearest(done)
            out = f"Do oddania masz: {polish_list([q.title for q in done[:4]])}."
            if near:
                out += f" Najbliżej oddasz {near[0].title}, {near[1].polish()}."
                self._set_topic("quest", near[0].title, near[0].quest_id)
            return out, (f"/wvt track {near[0].quest_id}" if near and self._waypoints_on() else "")
        if kind == "quest_nearest":
            near = book.nearest(todo) or book.nearest(quests)
            if not near:
                return "Żaden quest nie ma teraz znacznika na tej mapie.", ""
            self._set_topic("quest", near[0].title, near[0].quest_id)
            return f"Najbliżej masz {near[0].title}, {near[1].polish()}.", \
                (f"/wvt track {near[0].quest_id}" if self._waypoints_on() else "")
        target, named = self._pick_quest(polish, quests)
        if target is None:  # asks about something that is not in the quest log
            return self._lookup(polish) if kind == "quest_where" else ((self._knowledge(polish) or self._chat(polish)), "")
        self._set_topic("quest", target.title, target.quest_id)
        if kind == "quest_where":
            return self._where_text(target), (f"/wvt track {target.quest_id}" if self._waypoints_on() else "")
        if kind == "quest_progress":
            return self._progress(target) + ".", ""
        # anything else about quests: the LLM, with the quest log and everything about this quest as the only facts
        picked = [target] + [q for q in quests if q is not target]
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "")
        system += ("\nDziennik zadań gracza (to są jedyne fakty o questach, nie dodawaj niczego od siebie):\n"
                   + self._quest_facts(picked) + "\nPytanie dotyczy questa " + self._quest_details(target))
        out = self._llm_text([{"role": "system", "content": system}, *self._recent_history(),
                              {"role": "user", "content": polish}], max_tokens=300, limit=self._long_limit())
        return (out or self._progress(target) + "."), ""

    def _turnin_text(self, q: protocol.QuestEntry) -> tuple[str, str]:
        """Where the quest you were talking about is handed in."""
        world = self.world_getter()
        spot = world.finisher_spot(q.quest_id, self.quests.pos) if world is not None else None
        w = self.quests.where(q) if q.complete else None
        if spot is not None:
            where = w.polish() if w is not None else spot_where(world, spot, self.quests.pos)
            ready = "" if q.complete else " Najpierw dokończ cele: " + self._progress(q).split(": ", 1)[-1] + "."
            return f"{q.title} oddajesz u {spot.name}, {where}.{ready}", \
                (f"/wvt track {q.quest_id}" if self._waypoints_on() else "")
        return self._where_text(q), (f"/wvt track {q.quest_id}" if self._waypoints_on() else "")

    # ------------------------------------------------------------------ where next, places, waypoints
    def _player(self) -> Player:
        st, info = self.state, self.quests.player
        return Player(level=st.level if st else 0, race_id=info.race_id if info else 0,
                      class_id=info.class_id if info else 0, faction=info.faction_code if info else "",
                      done=self.quests.done, done_known=self.quests.done_known)

    def _waypoints_on(self) -> bool:
        return bool(self.cfg.get("assistant.auto_waypoint", True))

    def _line_hint(self, line: str) -> str:
        if line.startswith("/wvt way"):
            return " " + self._v("Znacznik czeka na Enter.", "Enter i masz pinezkę na mapie.",
                                 "Wciśnij Enter, to cię poprowadzę.")
        if line.startswith("/wvt track"):
            return " Enter włączy strzałkę."
        return ""

    def _advice(self, adv: Advice) -> tuple[str, str]:
        self._last_advice = adv
        if adv.quest_id:
            q = next((x for x in self.quests.quests() if x.quest_id == adv.quest_id), None) if self.quests.known else None
            self._set_topic("quest", q.title if q is not None else adv.text[:60], adv.quest_id)
        elif adv.item:
            self._set_topic("item", adv.item, advice=adv)
        elif adv.spot is not None:
            self._set_topic("place", adv.spot.name, advice=adv)
        line = adv.line if self._waypoints_on() else ""
        return adv.text + self._line_hint(line), line

    # ------------------------------------------------------------------ what to do now: a plan
    def _make_plan(self) -> Optional[Plan]:
        world, st = self.world_getter(), self.state
        try:
            return make_plan(world, self.quests, self._player(), self.quests.pos, st.xp_max if st else 0,
                             bool(st and st.in_group), self._chains_for(world) if world is not None else None)
        except Exception:
            log.exception("plan failed")
            return None

    def _show_step(self, step: Step) -> None:
        """The step she just named: the topic of the talk and what "zaprowadź mnie tam" means."""
        self._plan_shown = step.title
        if step.kind == "pickup":
            adv = Advice(f"{step.label()}, {step.where}", step.line)
            self._last_advice = adv
            self._set_topic("place", step.who or step.title, advice=adv)
        else:
            qid = step.quest_ids[0] if step.quest_ids else 0
            self._last_advice = Advice(step.title, step.line, quest_id=qid)
            self._set_topic("quest", step.title, qid)

    def _plan_answer(self, polish: str) -> tuple[str, str]:
        """"Co teraz robić?": the route through your quests, what is worth most and what to let go, in her words."""
        plan = self._make_plan()
        if plan is None or plan.empty:  # nothing in the log near you: new quests or the zone for your level
            return self._advice(where_next(self.world_getter(), self.quests, self._player(), self.quests.pos))
        first = (plan.route or plan.later)[0]
        self._show_step(first)
        self._sources = [f"plan: {len(plan.route)} steps, {len(plan.skip)} to let go, {len(plan.later)} later"]
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "") + PLAN + plan.facts()
        text = self._llm_text([{"role": "system", "content": system}, *self._recent_history(),
                               {"role": "user", "content": polish}], max_tokens=400, temperature=0.4,
                              limit=max(600, self._long_limit()),
                              thinking=str(self.cfg.get("assistant.plan_thinking", "low")))
        if text and plan.skip and not any(fold(s.title) in fold(text) for s, _ in plan.skip):
            # what to let go matters: Gemini left it out 4 times of 4 in the test, so it is added here
            text = text.rstrip(". ") + ". " + skip_sentence(*plan.skip[0])
        line = first.line if self._waypoints_on() else ""
        return (text or plan.text()) + self._line_hint(line), line

    def _plan_next(self, polish: str) -> tuple[str, str]:
        """"A potem?" / "co dalej?": the next step (the plan is made again: you moved, quests changed)."""
        plan = self._make_plan()
        if plan is None or not plan.route or not self._plan_shown:
            return self._plan_answer(polish)
        step = next((s for s in plan.route if s.title != self._plan_shown), None)
        if step is None:
            return "To już wszystko z tego planu. Zapytaj, co teraz robić, to ułożę nowy.", ""
        self._show_step(step)
        text = f"Teraz {step.label()}, {step.where}."
        if step.verdict == "great" and step.facts:
            text += " " + step.facts[0]
        line = step.line if self._waypoints_on() else ""
        return text + self._line_hint(line), line

    def _lookup(self, polish: str) -> tuple[str, str]:
        """'Gdzie jest X?' that is not about the quest log: services, dungeon entrances, NPCs, then the guides."""
        english = self._translate(polish, "pl-en")
        adv = find_place(self.world_getter(), polish, self._player(), self.quests.pos, english)
        if adv is not None:
            return self._advice(adv)
        return (self._knowledge(polish, english) or self._chat(polish)), ""

    def _item_where(self, polish: str) -> Optional[tuple[str, str]]:
        """'Gdzie kupię / skąd wezmę Rhapsody Malt?': who sells it, what drops it - the nearest first."""
        world = self.world_getter()
        if world is None or not hasattr(world, "find_item_in"):
            return None
        adv = find_item(world, polish, self._player(), self.quests.pos)
        if adv is None:
            return None
        if self.quests.known and any(fold(q.title) == fold(adv.item) for q in self.quests.quests()):
            return None  # "Beer Basted Boar Ribs" the quest, not the dish
        self._sources = [f"item: {adv.item}"]
        return self._advice(adv)

    _POINTERS = {"to", "tam", "tego", "ten", "te", "ta", "tu", "tutaj", "go", "ja", "je", "mi", "mnie", "na", "mapie",
                 "znacznik", "znacznika", "punkt", "pinezke", "prosze", "ej", "iskra", "postaw", "ustaw", "daj", "zrob",
                 "dodaj", "wstaw", "zaznacz", "pokaz", "wskaz", "zaprowadz", "doprowadz", "prowadz", "poprowadz",
                 "nawiguj", "do", "mnie", "waypoint", "marker", "jakis", "tego", "miejsca", "miejsce", "w"}

    def _waypoint(self, polish: str) -> tuple[str, str]:
        words = [w for w in fold(polish).split() if w not in self._POINTERS]
        if not words:
            adv = self._last_advice
            if adv and adv.line:
                return ("Stawiam znacznik, Enter." if adv.line.startswith("/wvt way") else "Włączam strzałkę, Enter."), \
                    adv.line
            return "Gdzie mam postawić znacznik? Powiedz na przykład: zaprowadź mnie do trenera.", ""
        english = self._translate(polish, "pl-en")
        if self.quests.known:
            scored = match_quests(f"{english} {polish}", self.quests.quests())
            if scored:
                q = scored[0][1]
                self._last_advice = Advice(q.title, f"/wvt track {q.quest_id}", quest_id=q.quest_id)
                return f"Prowadzę do {q.title}. Enter włączy strzałkę.", f"/wvt track {q.quest_id}"
        adv = find_place(self.world_getter(), polish, self._player(), self.quests.pos, english)
        if adv is None:
            return "Nie wiem, gdzie to jest. Spróbuj podać nazwę z gry.", ""
        if not adv.line:
            return adv.text, ""
        self._last_advice = adv
        return adv.text + " Znacznik czeka na Enter.", adv.line

    def _places(self) -> list[str]:
        world = self.world_getter()
        return list(world.names.values()) if world is not None else []

    def _reminder_add(self, polish: str) -> str:
        parsed = parse_reminder(polish, places=self._places())
        if parsed is None:
            return ("Kiedy mam przypomnieć? Powiedz na przykład: za dziesięć minut, o dwudziestej, na dwudziestym "
                    "poziomie albo w Ironforge.")
        r = self.reminders.add(parsed)
        what = f", {r.message}" if r.message else ""
        return self._v(f"Dobra, przypomnę ci {parsed.when_text}{what}.",
                       f"Zapisane: {parsed.when_text}{what}. Ja nie zapomnę, w przeciwieństwie do niektórych.",
                       f"Jasne, {parsed.when_text} dam ci znać{what}.")

    def _worth(self, polish: str) -> str:
        """'Czy warto robić ten quest?': the named quest, the one just taken, or the tracked one."""
        target = None
        if self.quests.known:
            quests = self.quests.quests()
            named = next((e.ref for e in self.last_correction.found if e.source == "questlog"), None)
            scored = [] if named else match_quests(f"{self._translate(polish, 'pl-en')} {polish}", quests)
            meant = None if named or scored else self._quest_by_meaning(polish, quests)
            topic = self._topic_quest(quests) if self._back(polish) else None
            if named:
                target = named
            elif scored:
                target = scored[0][1].quest_id
            elif meant is not None:
                target = meant.quest_id
            elif topic is not None:
                target = topic.quest_id
            elif self._last_accepted and time.time() - self._last_accepted[0] < 900:
                target = self._last_accepted[1]
            else:
                tracked = [q for q in quests if q.tracked and not q.complete]
                target = tracked[0].quest_id if tracked else None
        if target is None:
            return "O który quest chodzi? Powiedz nazwę albo weź go najpierw do dziennika."
        a = self._assess(target)
        if a is None:
            return "Tego questa nie mam w bazie, więc nie ocenię. Pewnie coś nowego z Forever."
        self._set_topic("quest", a.title, target)
        situation = f"Gracz pyta, czy warto robić questa {a.title}."
        facts = a.facts + [f"Ocena: {self._verdict_pl(a.verdict)}."]
        world = self.world_getter()
        if a.chain is not None and a.chain.length > 1 and world is not None:  # "co jest na końcu tego łańcucha?"
            facts.append("Questy łańcucha po kolei: " + " → ".join(world.quests[i]["n"] for i in a.chain.ids
                                                                   if i in world.quests) + ".")
        return self._persona_line(situation, facts, f"{a.title}: {a.text()}")

    _MINE_RE = re.compile(r"\b(podnios|zebral|zlootowal|lootowal|zgarn|wpadlo|mi wypadlo|wypadlo mi|moj|ostatni)")

    def _loot_answer(self, polish: str = "") -> str:
        """A roll that just started (need or greed), otherwise what you looted in the last minutes."""
        now = time.time()
        recent = [adv for t, adv in self._loot if now - t < 180]
        mine = bool(self._MINE_RE.search(fold(polish)))
        if recent and not mine:
            return " ".join(a.text for a in recent[-3:])
        picked = self._pickups_text()
        if picked:
            return picked
        return "W ostatnim kwadransie nic ci nie wpadło." if mine else "Nic ostatnio nie wypadło do losowania."

    _MONEY_RE = re.compile(r"(\d+)\s+(gold|silver|copper)", re.I)
    _LOOT_ITEM_RE = re.compile(r"\[([^\]]+)\](?:x(\d+))?")

    def _pickups_text(self, max_age_s: float = 900) -> str:
        counts: dict[str, int] = {}
        copper = 0
        for ln in self.chat.recent(80, kinds=LOOT_KINDS, max_age_s=max_age_s):
            if ln.kind == "MONEY":
                copper += sum(int(n) * {"gold": 10000, "silver": 100, "copper": 1}[u.lower()]
                              for n, u in self._MONEY_RE.findall(ln.text))
                continue
            for name, n in self._LOOT_ITEM_RE.findall(ln.text):
                counts[name] = counts.get(name, 0) + int(n or 1)
        if not counts and not copper:
            return ""
        parts = [f"{n} {plural(n, 'sztuka', 'sztuki', 'sztuk')} {name}" if n > 1 else name
                 for name, n in list(counts.items())[-8:]]
        text = f"W ostatnich {round(max_age_s / 60)} minutach wpadło ci: {polish_list(parts)}." if parts else ""
        if copper:
            g, rest = divmod(copper, 10000)
            s, c = divmod(rest, 100)
            money = polish_list([x for x in (f"{g} złota" if g else "", f"{s} srebra" if s else "",
                                              f"{c} miedzi" if c else "") if x])
            text = (text + f" Do tego {money} z łupów.") if text else \
                f"W ostatnich {round(max_age_s / 60)} minutach wpadło ci {money} z łupów."
        return text.strip()

    # ------------------------------------------------------------------ what you carry
    def _own_items(self) -> list["OwnItem"]:
        """Everything you carry or just looted, by name: stats from loot / the bag scan, counts from the bags."""
        out: dict[str, OwnItem] = {}
        with self._items_lock:
            stats = dict(self._items)
        for key, (when, it) in stats.items():
            out[key] = OwnItem(it.name, it.class_id, it.subclass_id, it.equip_loc, it.quality, it.count, loot=it,
                               looted_at=when)
        for b in self.bags.items:
            o = out.get(b.name.lower())
            if o is not None:
                o.bag, o.count = b, b.count
            else:
                out[b.name.lower()] = OwnItem(b.name, b.class_id, b.subclass_id, b.equip_loc, b.quality, b.count,
                                              bag=b)
        world = self.world_getter()
        for ln in self.chat.recent(80, kinds=LOOT_KINDS, max_age_s=1800):
            for name, _ in self._LOOT_ITEM_RE.findall(ln.text):
                o = out.get(name.lower())
                if o is None:
                    rec = None
                    if world is not None and hasattr(world, "_item_names"):
                        ids = world._item_names.get(fold(name)) or []
                        rec = world.items.get(ids[0]) if ids else None
                    out[name.lower()] = OwnItem(name, rec.get("c", -1) if rec else -1, rec.get("sc", -1) if rec else -1,
                                                "", -1, 1, looted_at=ln.when)
                elif o.looted_at is None:
                    o.looted_at = ln.when
        return list(out.values())

    _GENERIC_ITEM = re.compile(r"\b(rzecz\w*|przedmiot\w*|item\w*|bron\w*|cos)\b")

    def _find_own(self, text: str) -> tuple[list["OwnItem"], bool, bool]:
        """(the items the question is about, best first; whether one was named; whether the first one is surely
        it: named, its kind said ("kij"), "ta rzecz" (the newest loot) or Google's vectors sure of it)."""
        own = self._own_items()
        folded = fold(text)
        if not own:  # nothing known yet: say so only when it is clearly about an item
            return [], False, bool(self._GENERIC_ITEM.search(folded) or any(re.search(rx, folded)
                                                                            for rx, *_ in _ITEM_KINDS))
        found = {fold(e.name) for e in self.last_correction.found if e.kind in ("item", "word")}
        named = [o for o in own if fold(o.name) in found or (len(o.name) >= 4 and fold(o.name) in folded)]
        if named:
            return named, True, True
        cands = own
        sure = bool(self._GENERIC_ITEM.search(folded))
        for rx, cls, subs, locs in _ITEM_KINDS:
            if re.search(rx, folded):
                sure = True
                cands = [o for o in cands if (o.class_id == cls and (not subs or o.subclass_id in subs)
                                              and (not locs or o.equip_loc in locs))]
                break
        for rx, q in _QUALITY_WORDS:
            if re.search(rx, folded):
                cands = [o for o in cands if o.quality == q] or cands
                break
        if len(cands) > 1 or not sure:
            order: dict[str, int] = {}
            trusted = False
            if self.lexicon is not None:
                hits = self.lexicon.search(text, k=12, kinds={"item"}, sources={"loot", "bag"})
                order = {fold(e.name): i for i, (e, _) in enumerate(hits)}
                trusted = bool(hits) and self.lexicon.trusted(hits[0][1], "item")
            if trusted:  # Google's vectors know "co to za ząb?" (11 of 12 right in the bench)
                cands.sort(key=lambda o: (order.get(fold(o.name), 99), -(o.looted_at or 0)))
                sure = True
            else:  # the local e5 only breaks ties: the newest loot first ("ten kij, który znalazłem")
                cands.sort(key=lambda o: (-(o.looted_at or 0), order.get(fold(o.name), 99)))
        return cands, False, sure

    def _item_names_in(self, text: str, include_topic: bool = True) -> list[str]:
        """Item names the question names: brackets, corrected names, your bags, the world index, the item topic."""
        world = self.world_getter()
        folded = fold(text or "")
        raw: list[str] = list(re.findall(r"\[([^\]]+)\]", text or ""))
        for e in self.last_correction.found:
            if (e.kind == "item" or e.source in ("bag", "loot")) and getattr(e, "name", ""):
                raw.append(e.name)
        for it in self.bags.items:
            if len(it.name) >= 4 and fold(it.name) in folded:
                raw.append(it.name)
        names_map = getattr(world, "_all_item_names", {}) if world is not None else {}
        for key in names_map:
            if len(key) >= 5 and key in folded:
                raw.append(key)
        if include_topic:
            t = self._topic_now()
            if t is not None and t.kind == "item" and t.name:
                raw.append(t.name)
        out: list[str] = []
        seen: set[str] = set()
        for name in raw:
            canon = name
            rec = None
            if world is not None and hasattr(world, "item_record"):
                try:
                    rec = world.item_record(name)
                except Exception:
                    rec = None
            if isinstance(rec, dict) and rec.get("n"):
                canon = rec["n"]
            key = fold(canon).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(canon)
            if len(out) >= 4:
                break
        return out

    def _item_intent(self, text: str) -> Optional[str]:
        """item_compare / item_about when the question names an item and asks about it, else None."""
        if not self.cfg.get("assistant.item_help", True):
            return None
        folded = fold(text or "")
        explicit = self._item_names_in(text, include_topic=False)
        if not explicit and _QUEST_WORD_RE.search(folded):
            return None  # "czy warto robić ten quest?" stays a quest question, even after an item talk
        names = explicit or self._item_names_in(text)
        if not names:
            return None
        if len(names) >= 2 and _ITEM_COMPARE_RE.search(folded):
            return "item_compare"
        if _ITEM_QUESTION_RE.search(folded):
            return "item_about"
        return None

    _ITEM_CLASS_LABEL = {0: "jedzenie lub napój", 2: "broń", 4: "pancerz", 5: "odczynnik", 7: "materiał",
                         9: "przepis", 12: "przedmiot questowy"}
    _ITEM_QUALITY_LABEL = {0: "szary", 1: "biały", 2: "zielony", 3: "niebieski", 4: "fioletowy", 5: "legendarny"}

    def _item_facts(self, name: str, own: Optional["OwnItem"] = None) -> str:
        """The item's facts as lines: kind, levels, your copy, vendors, drops, quests. No LLM, no prices."""
        world = self.world_getter()
        record = None
        if world is not None and hasattr(world, "item_record"):
            try:
                record = world.item_record(name)
            except Exception:
                record = None
        if not isinstance(record, dict):
            record = None
        if record is None and own is None:
            self._sources = [f"item facts: {name}"]
            self._set_topic("item", name)
            return "Nie mam tego przedmiotu w bazie świata; nie znam receptur ani cen."
        shown = (record.get("n") if record else "") or name
        p = self._player()
        cls = record.get("c") if record else (own.class_id if own is not None else None)
        lines = [shown]
        label = self._ITEM_CLASS_LABEL.get(cls, f"klasa {cls}") if isinstance(cls, int) else ""
        if label:
            lines.append(f"Rodzaj: {label}.")
        level = record.get("il") if record else (own.loot.item_level if own is not None and own.loot else None)
        if isinstance(level, int) and level > 0:
            lines.append(f"Poziom przedmiotu: {level}.")
        needed = record.get("rl") if record else (own.loot.min_level if own is not None and own.loot else None)
        if isinstance(needed, int) and needed > 0:
            lines.append(f"Wymagany poziom: {needed}.")
        if own is not None:
            quality = self._ITEM_QUALITY_LABEL.get(own.quality, f"jakość {own.quality}")
            lines.append(f"Jakość: {quality}. Masz: {own.count}.")
        if record:
            vendors = record.get("v") or []
            if vendors:
                lines.append("Sprzedawcy: " + polish_list(list(vendors)[:5]) + ".")
            drops = list(record.get("nd") or []) + list(record.get("od") or [])
            if drops:
                lines.append("Dropi: " + polish_list(drops[:6]) + ".")
            quests = list(record.get("sq") or []) + list(record.get("rq") or [])
            if quests:
                lines.append("Questy: " + polish_list(list(dict.fromkeys(quests))[:5]) + ".")
        if own is not None and own.loot is not None:
            lines.append(item_verdict(own.loot, p.class_id, p.level))
        lines.append("Brak danych o recepturach i reagentach.")
        lines.append("Brak danych o cenach rynkowych.")
        self._sources = [f"item facts: {shown}"]
        self._set_topic("item", shown)
        return "\n".join(lines)

    def _item_compare(self, text: str) -> tuple[str, str]:
        """Two items side by side from their facts; never from market prices (there are none)."""
        names = self._item_names_in(text)
        if len(names) < 2:
            return "Podaj dwa przedmioty, np. {item_compare} [A] i [B].", ""
        first, second = names[0], names[1]
        a = self._item_facts(first)
        b = self._item_facts(second)
        self._sources = [f"item facts: {first}", f"item facts: {second}"]
        self._set_topic("item", second)
        return (f"{a} {b} Nie mam cen rynkowych, więc nie powiem, który jest więcej warty."), ""

    def _item_about(self, text: str) -> Optional[str]:
        """None: not clearly about one of your items ("co to za miejsce?") - her LLM answers from the facts."""
        cands, named, sure = self._find_own(text)
        if not self.cfg.get("assistant.item_help", True):
            if not sure:
                return None
            if not cands:
                if not self.bags.known and not self._items:
                    return "Jeszcze nie widzę twoich toreb ani łupów. Daj mi chwilę po zalogowaniu albo zrób /reload."
                return "Nie widzę takiego przedmiotu w twoich łupach ani w torbach."
            o = cands[0]
            p = self._player()
            answer = item_verdict(o.loot, p.class_id, p.level) if o.loot is not None else self._item_plain(o)
            same = (o.class_id, o.subclass_id, o.equip_loc)  # "masz też" only for items of the same kind
            others = [c.name for c in cands[1:]
                      if c.name != o.name and (c.class_id, c.subclass_id, c.equip_loc) == same][:2]
            if others and not named:
                answer += f" Jeśli chodziło ci o inny, masz też {polish_list(others)}."
            self._sources = [f"own item: {o.name}" + (" (stats)" if o.loot is not None else "")]
            self._set_topic("item", o.name)
            return answer
        if cands and (named or sure):
            return self._item_facts(cands[0].name, cands[0])
        if sure and not cands:
            if not self.bags.known and not self._items:
                return "Jeszcze nie widzę twoich toreb ani łupów. Daj mi chwilę po zalogowaniu albo zrób /reload."
            return "Nie widzę takiego przedmiotu w twoich łupach ani w torbach."
        names = self._item_names_in(text)
        if not names:
            return None
        name = names[0]
        own = next((o for o in self._own_items() if fold(o.name) == fold(name)), None)
        return self._item_facts(name, own)

    def _item_plain(self, o: "OwnItem") -> str:
        """An item without stats: what it is for."""
        world = self.world_getter()
        if world is not None and self.quests.known:
            for q in self.quests.quests():
                rec = world.quests.get(q.quest_id) or {}
                names = [world.items.get(r[0], {}).get("n", "") for r in rec.get("loot") or [] if r]
                names += [re.sub(r"^\s*\d+\s*/\s*\d+\s+", "", t) for _, t in q.objectives]
                if any(fold(n) == fold(o.name) for n in names if n):
                    return f"{o.name} jest do questa {q.title}."
        if o.quality == 0:
            return f"{o.name} to szary śmieć. Sprzedaj u najbliższego vendora."
        if o.class_id == 0:
            return f"{o.name} jest do zjedzenia albo wypicia, najlepiej po walce."
        if o.class_id == 7:
            return f"{o.name} to materiał do rzemiosła. Przyda się w zawodzie albo sprzedasz go na aukcji."
        if o.class_id == 12:
            return f"{o.name} to przedmiot do questa."
        return f"To {o.name}. Nie mam jego statystyk, więc nie porównam."

    @staticmethod
    def _count_text(it) -> str:
        return f"{it.count} {it.name}" if it.count > 1 else it.name

    def _bags_unknown(self) -> str:
        return "Jeszcze nie widzę twoich toreb. Daj mi chwilę po zalogowaniu albo zrób /reload."

    def _bag_list(self) -> str:
        if not self.bags.known:
            return self._bags_unknown()
        items = self.bags.items
        if not items:
            return "Torby masz puste."
        objectives = {fold(e.name) for e in self.lexicon.entries() if e.source == "objective"} \
            if self.lexicon is not None else set()
        quest = [it for it in items if fold(it.name) in objectives or it.class_id == 12]
        gear = [it for it in items if it.equip_loc and it.quality >= 1 and it not in quest]
        junk = [it for it in items if it.junk and it not in quest]
        rest = sorted((it for it in items if it not in quest and it not in gear and it not in junk),
                      key=lambda it: (-it.quality, -it.count))
        n = len(items)
        parts = [f"W torbach masz {n} {plural(n, 'rzecz', 'rzeczy', 'rzeczy')}."]
        if quest:
            parts.append("Do questów: " + polish_list([self._count_text(i) for i in quest[:4]]) + ".")
        if gear:
            parts.append("Do założenia: " + polish_list([i.name for i in gear[:3]]) + ".")
        if rest:
            parts.append("Poza tym " + polish_list([self._count_text(i) for i in rest[:5]]) + ".")
        if junk:
            parts.append(f"I {len(junk)} {plural(len(junk), 'szary śmieć', 'szare śmieci', 'szarych śmieci')} do sprzedania.")
        self._sources = [f"bags: {n} items"]
        return " ".join(parts)

    def _bag_junk(self) -> str:
        if not self.bags.known:
            return self._bags_unknown()
        junk = self.bags.junk()
        if not junk:
            return "Nie masz szarych śmieci, czysto."
        n = sum(it.count for it in junk)
        more = " I jeszcze inne." if len(junk) > 5 else ""
        return (f"Do sprzedania masz {n} {plural(n, 'szary śmieć', 'szare śmieci', 'szarych śmieci')}: "
                f"{polish_list([self._count_text(i) for i in junk[:5]])}.{more}")

    def _bag_count(self, text: str, quiet: bool = False) -> str:
        """"Czy mam Rhapsody Malt?" / "ile mam Linen Cloth?" from the bags; "" when no item of yours is named
        (``quiet``: also when the item is a quest objective - the quest log answers that better)."""
        if not self.bags.known:
            return "" if quiet else self._bags_unknown()
        folded = fold(text)
        items = self.bags.items
        found = [e for e in self.last_correction.found if e.kind == "item"]
        if quiet and any(e.source in ("objective", "questlog") for e in found):
            return ""
        target = next((e.name for e in found), None) or \
            next((it.name for it in items if len(it.name) >= 4 and fold(it.name) in folded), None)
        if target is None and not quiet and self.lexicon is not None:  # "ile mam materiału?" (Google's vectors)
            hits = self.lexicon.search(text, k=1, kinds={"item"}, sources={"bag"})
            if hits and self.lexicon.trusted(hits[0][1], "item"):
                target = hits[0][0].name
        if target is None:
            return ""  # her LLM answers it: the facts list your bags
        n = self.bags.count(target)
        self._sources = [f"bags: {target}"]
        if n:
            return f"Masz {n} {plural(n, 'sztukę', 'sztuki', 'sztuk')} {target}." if n > 1 else f"Masz {target}, jedną sztukę."
        return f"Nie masz {target} w torbach."

    # ------------------------------------------------------------------ guides (local knowledge base)
    _ABILITY_RE = re.compile(r"\b(umiejetn\w*|skill\w*|spell\w*|czar\w*|zaklec\w*|talent\w*|trener\w*|naucz\w*|zdolnosc\w*|rotacj\w*)\b")
    _MY_LEVEL_RE = re.compile(r"\b(moj\w* poziom\w*|moim poziomie|moj\w* level\w*|moim levelu|na moj\w*|dla mnie)\b")

    _GAME_WORDS = re.compile(
        r"\b(quest\w*|kwest\w*|zadani\w*|dung\w*|loch\w*|instancj\w*|raid\w*|rajd\w*|boss\w*|mob\w*|npc|potwor\w*"
        r"|przedmiot\w*|item\w*|bron\w*|zbroj\w*|pancerz\w*|skill\w*|umiejetn\w*|zdolnosc\w*|talent\w*|spell\w*"
        r"|czar\w*|zaklec\w*|klas\w*|ras\w*|profesj\w*|zawod\w*|level\w*|poziom\w*|lvl|stref\w*|kontynent\w*|frakcj\w*"
        r"|przymierz\w*|alliance|hord\w*|trener\w*|wierzchow\w*|mount\w*|eliksir\w*|mikstur\w*|przepis\w*|receptur\w*"
        r"|reputacj\w*|frakcj\w*|gildi\w*|pvp|arena\w*|battleground\w*|zlot\w*|miasto|miasta|wysp\w*|lore|historia"
        r"|forever|skyborne|azeroth|kalimdor|wow|warcraft|gra|grze|grac|tank\w*|heal\w*|dps|aggro|loot\w*|drop\w*"
        r"|craft\w*|farm\w*|exp\w*|karczm\w*|odpoczyn\w*|rested|hearth\w*|kamien\w*)\b")
    _PROPER_NOUN = re.compile(r"(?<=[\w,] )[A-Z][a-z]*[A-Z]?[a-z']+")

    def _is_game_question(self, polish: str, english: str) -> bool:
        """Small talk must not turn into 'I don't know' from the guides: a game word, or a name from the game."""
        return bool(self._GAME_WORDS.search(fold(polish)) or self._PROPER_NOUN.search(english or ""))

    def _knowledge(self, polish: str, english: Optional[str] = None) -> str:
        kb = self.kb_getter()
        if kb is None:
            return ""
        floor = float(self.cfg.get("assistant.kb_min_score", 0.8))
        try:
            quick = kb.search(polish, "", k=1)
        except Exception:
            log.exception("knowledge base search failed")
            return ""
        if not quick or quick[0].cosine < floor - 0.03:
            return ""  # nothing close at all
        english = self._translate(polish, "pl-en") if english is None else english
        if not self._is_game_question(polish, english):
            return ""
        p = self._player()
        query = english
        folded = fold(polish)
        if self._ABILITY_RE.search(folded) and p.class_id in CLASS_EN:
            query += f" {CLASS_EN[p.class_id]} abilities (Forever)"
        if p.level and self._MY_LEVEL_RE.search(folded):
            query += f" level {p.level}"
        hits = kb.search(polish, query, k=4)
        if not hits or max(h.cosine for h in hits) < floor:
            return ""
        context = "\n\n".join(f"[{h.title}{' > ' + h.section if h.section else ''}]\n{h.text[:900]}" for h in hits)
        extra = str(self.cfg.get("assistant.persona_extra", "") or "")
        system = PERSONA.format(name=self.name, extra=(" " + extra) if extra else "") + KNOWLEDGE
        who = self._context() + (f", klasa {CLASS_EN[p.class_id]}" if p.class_id in CLASS_EN else "")
        facts = self._facts(polish)
        topic = self._topic_text()
        system += f"Gracz: {who}.\n" + (f"W grze teraz: {facts}.\n" if facts else "") + \
            (f"{topic}\n" if topic else "") + f"Fragmenty:\n{context}"
        out = self._llm_text([{"role": "system", "content": system}, *self._recent_history(),
                              {"role": "user", "content": polish}], max_tokens=300, temperature=0.3,
                             limit=self._long_limit())
        self._sources += [h.title + (f" > {h.section}" if h.section else "") for h in hits[:4]]
        log.info("knowledge answer from %s", ", ".join(h.title for h in hits[:3]))
        return out

    def _info(self, kind: str) -> str:
        st = self.state
        if st is None or time.time() - self.state_at > 120:
            return "Nie widzę danych z gry. Sprawdź, czy dodatek jest włączony: wpisz /wvt asystent on."
        if kind == "gold":
            silver, copper = divmod(st.silver_copper, 100)
            return f"Masz {st.gold} złota, {silver} srebra i {copper} miedzi."
        if kind == "xp":
            if st.xp_max <= 0:
                return f"Jesteś na poziomie {st.level}, dalej już się nie da."
            need = max(0, st.xp_max - st.xp)
            pct = round(100 * st.xp / st.xp_max)
            return (f"Poziom {st.level}, masz {pct} procent paska. Do następnego brakuje {need} doświadczenia."
                    + self.session.eta_text(st))
        if kind == "where":
            return f"Jesteś w {st.zone}" + (f", dokładnie {st.subzone}." if st.subzone and st.subzone != st.zone else ".")
        if kind == "bags":
            return f"Masz {st.bag_free} wolnych miejsc na {st.bag_slots}."
        if not st.in_group:
            return "Jesteś solo. Mam kogoś zaprosić?"
        if st.in_raid:
            return f"Jesteś w raidzie, razem {st.group_size} osób."
        others = self._short(st.members)
        return f"W party masz {polish_list(others)}." if others else "Jesteś w grupie, ale nie widzę imion."

    # ------------------------------------------------------------------ output
    def _respond(self, reply: str, line: str, insert_status: str, event: str = "comment") -> None:
        """Show and speak an answer; ``event`` names a spontaneous one for the journal ("" = a question)."""
        if event and (reply or line):
            self._journal({"event": event, "reply": reply, "line": line})
            self.transcript.appendleft({
                "at": time.time(), "heard": "", "fixed": "", "reply": reply, "line": line,
                "intent": f"event:{event}", "route": "event", "llm": dict(self._last_llm),
                "sources": list(self._sources), "ms": 0, "event": event,
            })
        if event and reply:  # her own comment is part of the talk ("a co jest na końcu?" right after it)
            self._history.append({"role": "assistant", "content": reply, "at": time.time()})
        footer = ""
        if line:
            footer = {"inserted": "W czacie – Enter wykonuje, Esc anuluje",
                      "copied": "WoW nie był aktywny – skopiowane do schowka"}.get(insert_status, insert_status)
        lines = [f"{self.name}: {reply}"] if reply else []
        if line:
            lines.append(line)
        if reply or line:
            self.ui.status("ok", footer or self.name, lines, float(self.cfg.get("hud.show_seconds", 6)) + 2)
        if reply:
            self._speak(reply)

    def _speak(self, text: str) -> None:
        voice = self.voice_getter()
        voice_id = str(self.cfg.get("assistant.voice_id", "") or "")
        if not voice_id and voice is not None:
            voice_id = str(getattr(voice, "default_voice_id", "") or "")
        if voice is None or not voice_id:
            if not self._warned_no_voice:
                self._warned_no_voice = True
                log.info("assistant has no voice yet (assistant.voice_id or ElevenLabs missing) - text only")
            return
        voice.say(text, voice_id, "pl", float(self.cfg.get("assistant.volume", 1.0)), speaker=self.name)


@dataclass
class Topic:
    """What you and she are talking about, so "a gdzie to?", "ile za niego expa?" need no name."""

    kind: str  # quest | item | place
    name: str
    quest_id: int = 0
    advice: Optional[Advice] = None
    at: float = field(default_factory=time.time)


@dataclass
class OwnItem:
    """An item you carry or just looted, put together from loot (stats), the bags (count) and loot lines."""

    name: str
    class_id: int
    subclass_id: int
    equip_loc: str
    quality: int
    count: int = 1
    loot: Optional[protocol.LootItem] = None  # with stats: looted, or a wearable from the bag scan
    bag: object = None  # wvt.bags.BagItem
    looted_at: Optional[float] = None


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def clean_reply(text: str, limit: int = 240) -> str:
    """One or two plain spoken sentences: no markup, emoji, quotes or a half sentence at the end."""
    t = re.sub(r"<[^>]+>|\*+|#+|`+", "", text or "")
    t = _EMOJI_RE.sub("", t)
    t = " ".join(t.replace("\n", " ").split()).strip(" \"'„”")
    if len(t) > limit:
        cut = t[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        t = cut[: end + 1] if end > 40 else cut.rsplit(" ", 1)[0] + "…"
    return t
