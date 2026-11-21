# WoW Voice Translator

A voice chat translator for **World of Warcraft Forever**, made for Polish players: you speak Polish, and English text appears in the chat box.

- **Push-to-talk translation.** Hold a key and speak Polish. The English translation is pasted into the WoW chat box, and you send it yourself with Enter.
- **Hover translation.** Hover the mouse over an English chat line to see it in Polish.
- **WoW chat slang.** It understands slang such as mb, wc, ty, LFM and WTS.
- **Local and offline.** Speech recognition and translation run on your own PC and GPU, with no internet, account or fees.
- **Character voices (optional).** `/say` and `/party` lines are read aloud with a voice that fits each character's race and sex (ElevenLabs, paid). `/say` gets quieter the farther away the speaker stands.
- **Voice assistant "Iskra" (optional).** On a separate key you talk to her in Polish:
  - She invites the players who asked for an invite and summarises chat and whispers.
  - She knows your quest log, plans what to do next and sets map waypoints.
  - She compares loot with what you wear and answers game questions from a local knowledge base.

> The app's window, its on-screen messages and the assistant are in Polish. The chat you send is always English.

## Requirements

- Windows 10 or 11, 64-bit.
- **An NVIDIA GPU** (RTX series) with a current driver. It was tested on an RTX 5080 with 16 GB.
  - Speech recognition and translation run on the GPU next to the game.
  - When there isn't enough free VRAM, speech recognition falls back to the CPU.
- **[Python 3.12](https://www.python.org/downloads/).** Keep *py launcher* ticked in the installer. You don't need the CUDA toolkit, because the Python packages bring the CUDA libraries.
- About **12 GB** of free disk space. Setup downloads about 8 GB.
- A microphone.
- WoW Forever, installed with Battle.net.

Optional:
- an [ElevenLabs](https://elevenlabs.io) account for character voices and the assistant's voice;
- a [Google AI Studio](https://aistudio.google.com) key for a smarter assistant;
- MongoDB for voice assignments. Without it they are kept in local files.

## Installation (once)

1. **Get the code.** Either clone it:

   ```
   git clone https://github.com/matheowis/wow-voice-translator.git
   ```

   or use **Code → Download ZIP** on GitHub and unzip it, for example to `C:\WoWVoiceTranslator`. Avoid folders that need admin rights, such as `Program Files`.

2. **Open PowerShell in that folder.** In File Explorer, click the address bar, type `powershell` and press Enter.

3. **Run the setup script:**

   ```
   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
   ```

   The script:
   - creates a virtual environment `.venv` with Python 3.12 (`py -3.12`) and installs `requirements.txt`;
   - downloads the models into `models\` and llama.cpp into `runtime\` (about 8 GB, resumable, every file SHA-256 verified):
     - Whisper large-v3-turbo (ONNX), for speech recognition;
     - Parakeet TDT 0.6B v3 (ONNX), the fallback recognizer;
     - NVIDIA Nemotron 3.5 Bolt E4B (GGUF), the translator;
     - llama.cpp b12177 for CUDA, which runs the translator;
   - installs the addon into `<WoW folder>\_forever_\Interface\AddOns\WoWVoiceTranslator`. It finds the game through the Battle.net registry entry;
   - creates `.env` from `.env.example`. That is where optional keys go.

   If a download fails, run the script again; it continues where it stopped. Options:

   | Option | Meaning |
   |---|---|
   | `-Wow "D:\Games\World of Warcraft"` | the WoW folder (the one that contains `_forever_`), when it isn't found automatically |
   | `-SkipAddon` | don't install the addon |
   | `-Python "py -3.12"` | how to start Python 3.12 |

4. **Check the installation** (optional). This prints what is installed and what is missing:

   ```
   .venv\Scripts\python -m wvt.cli check
   ```

## Starting

1. **Start the app.** Double-click **`scripts\run.bat`**. An icon appears in the system tray next to the clock.
   - Loading the models takes 1–2 minutes, or several minutes from a slow (USB) drive.
   - Double-click the icon to open the window. `scripts\run.bat --show` opens it right away.
   - The first start creates `config.json` with all settings.
2. **Set the display mode.** Start WoW Forever and, in **Options → Graphics**, set **Display Mode: Windowed (Fullscreen)**.
   - The addon talks to the app through a small grid of coloured squares in the top-left corner of the game.
   - The app reads that grid with an ordinary screen capture, and only while WoW is the active window.
   - In plain *Windowed* mode the grid can come out scaled, and exclusive *Fullscreen* can hide it from screen capture. *Windowed (Fullscreen)* works.
3. **Enable the addon.** At the character selection screen, click **AddOns** and make sure **WoW Voice Translator** is enabled.
4. **Test the connection.** In the game, type `/wvt test`. The app shows **„✓ połączenie z addonem działa”** ("the connection to the addon works") in the middle of the screen.

## Using it

| Key | What it does |
|---|---|
| hold **`** (backtick, under Esc) | speak Polish → English text for **/say** |
| hold **Caps Lock** | speak Polish → English text for **/party** |
| hold **mouse button 5** (side button) | talk to the assistant |
| **Esc** while speaking | cancel |
| **Pause** | przerwij odczyt – stop the reader: Iskra's answer and any character voice (global, also while Iskra speaks) |
| **Enter** | send the text. The app never sends anything by itself: it only pastes into the chat box. |

- **Hover translations.** Hover the mouse over a chat line for a moment and the Polish translation appears above it.
- **Confirm before inserting (optional).** With **Settings → Push-to-talk → "Pokaż tekst do edycji przed wstawieniem (Enter wstawia, Esc anuluje)"** checked, a small always-on-top box shows the whole chat line (for example `/p hello`) after push-to-talk instead of pasting it at once. **Enter** (or **Wstaw**) inserts it into the chat box, **Esc** (or **Anuluj**) throws it away; you still press Enter in WoW to send, as usual. The box counts the 255-byte limit (`translation.max_chat_chars`) and truncates anything longer when inserted. Iskra's own commands are unaffected: they already land in the chat box and run only on Enter.
- **Settings.** Keys, microphone and other settings are in the app window. The keys react only while WoW is the active window. Under the push-to-talk table, **Klawisz przerwania odczytu** sets the global stop key (`voice.stop_key`, default `pause`; `—` turns it off). Right below it, **Klawisz historii Iskry** sets the key that shows/hides the history overlay (`assistant.history_key`, default `insert`; `—` turns it off).

Addon commands in the game:

| Command | What it does |
|---|---|
| `/wvt test` | test the connection to the app |
| `/wvt hdr on/off` | HDR-safe grid: darker cell levels for HDR displays (try this first if the grid is visible but `/wvt test` fails) |
| `/wvt on`, `/wvt off` | hover translations on/off |
| `/wvt say on/off`, `/wvt party on/off` | read /say or /party aloud (character voices, off by default) |
| `/wvt lang en`, `/wvt lang pl` | hear other players in English, or translated to Polish |
| `/wvt panel` | show/hide the voice panel above the chat (Shift-drag moves it) |
| `/wvt voice` | voice status: what is on, how many lines were sent or skipped |
| `/wvt voicedebug on` | print the race, sex and measured distance for every line that is read |
| `/wvt asystent off` | stop sending chat and game state to the assistant |

## Character voices (optional, ElevenLabs)

1. Put your key in `.env`: `ELEVENLABS_API_KEY=...`. The `.env` file is git-ignored; never commit it.
2. Create the voices once, in your own ElevenLabs account. The default is 90 voices: 9 races × 2 sexes × 5 voices each.

   ```
   .venv\Scripts\python -m wvt.voice.design
   ```

   With a smaller plan, use `--slots 1` or `--slots 2`.
3. Restart the app. In the game, switch on **Say** and/or **Party** on the panel above the chat. They are off by default, so nothing is spent until you do.

The full guide is in **[docs/VOICES.md](docs/VOICES.md)**: permissions for the key, credits, re-rolling voices you don't like, rating voices, the assistant's voice and how `/say` volume follows distance.

### No ElevenLabs? Local voice for Iskra

Without `ELEVENLABS_API_KEY` the assistant can still speak, using a local engine. Character voices (`/say`, `/party`) stay text-only in this mode.

- `voice.backend` (default `auto`): use ElevenLabs when the key is set, otherwise the local voice. `local` always uses the local engine, `elevenlabs` never falls back.
- Settings -> **Głos Iskry (gdy brak ElevenLabs)**: the checkbox turns the local voice on/off, the list picks **Automatycznie (Piper -> SAPI)**, **Piper** or **SAPI (Microsoft Paulina)**. Changes need an app restart only.
- **Piper** (optional, GPL): `pip install -r requirements-tts.txt`, then `python -m wvt.cli tts --download`. Check it with `python -m wvt.cli tts --test "Dzień dobry"`.
- **SAPI**: without Piper, the built-in Windows voice (Microsoft Paulina, `pl-PL`) is used. Nothing to install.
- No addon reinstall is needed; the Lua addon still only draws text.

### Speech recognition: local or cloud

By default speech recognition runs locally on your GPU (Whisper large-v3-turbo, Parakeet as the fallback). Nothing leaves your PC, and this stays the default.

You can switch to a cloud provider instead, e.g. on a machine without a fast GPU:

- **Settings -> Rozpoznawanie mowy (STT)**: pick **Cloud (API OpenAI-compatible)**, set the base URL, the model and the name of the key variable, and paste the key (it is written to `.env`; leaving the field empty keeps the current key).
- Any provider with an OpenAI-compatible `/audio/transcriptions` endpoint works. For Groq use `https://api.groq.com/openai/v1` with `whisper-large-v3-turbo` and `GROQ_API_KEY`; OpenAI's own API and OpenRouter also work when their audio endpoint serves the model you pick.
- **Audio is sent to the provider** (Groq/OpenAI/OpenRouter) for every utterance; nothing about the audio or the transcript is logged locally.
- Changes need an app restart (`stt.backend`, `stt.cloud.*`).

The equivalent `config.json` keys are described in [docs/SZCZEGOLY.md](docs/SZCZEGOLY.md).

### Who speaks and writes in which language

| What | Language | How to change |
|---|---|---|
| What you send to chat (` and Caps Lock) | always **English**: you speak Polish, the app translates | – |
| Hover translation | **Polish** | `/wvt on` / `/wvt off` |
| Other players' voices (/say, /party) | **EN**: the English original, or **PL**: translated to Polish | **Glos EN/PL** button on the panel, `/wvt lang pl` / `/wvt lang en` |
| Your own character | repeats your line in its voice: in English in EN mode, in your own Polish words in PL mode | as above; turn it off in the app's **Voices** tab |
| The assistant | always **Polish** | – |

## The assistant (Iskra, Polish only)

Hold **mouse button 5** and say what you want in Polish.
- She answers aloud with an ElevenLabs voice (`assistant.voice_id`) or, without a key, with the local voice (Piper/SAPI - see above). With both off she answers in text on screen.
- When something has to happen in the game, she types a ready command into the chat box, for example `/wvt inv Ann Bob`. **It runs only when you press Enter**; Esc cancels.
- **Review the transcript first (optional).** With **Settings → Push-to-talk → "Pokaż tekst do edycji przed wysłaniem do Iskry (Enter zatwierdza, Esc anuluje)"** (or `"assistant": {"confirm_transcript": true}`, the default) a small always-on-top box shows what she heard after mouse5. Fix it and press **Enter** to let Iskra process it, or **Esc** / **Anuluj** to drop it. This is separate from the chat-line confirm above: that one edits the final line, this one edits the Polish transcript.

| Say, for example | What happens |
|---|---|
| „zaproś ich”, „weź go doproś do party”, „zaproś Artasa” | invites everyone who wrote "inv me" lately, the last player who asked or spoke, or a named player (Polish declensions are understood) |
| „wyrzuć Bolka”, „daj lidera Jainie”, „zrób raid”, „ready check” | group commands; the game blocks some of them in combat |
| „co mi napisał?”, „co piszą na trade?”, „kto szuka grupy?” | reads whispers and summarises chat channels in Polish |
| „odpisz mu, że zaraz będę” | translates your reply to English and types it into the chat box |
| „co teraz robić?”, „a potem?” | a plan for now: the shortest route through your quests, what matters most, what to skip |
| „czy warto robić ten quest?”, „gdzie go oddać?” | quest advice: XP, chain, reward; she remembers what you are talking about for 15 minutes |
| „co jest do wzięcia w tej strefie?”, „jakie questy mogę wziąć?” | quests to take in your current zone from QuestieDB: level, faction, class and completed quests checked, nearest starter first |
| „gdzie jest trener magów?”, „zaprowadź mnie tam” | nearest NPC or place, plus a map waypoint |
| „czy ten kij, który znalazłem, jest lepszy?”, „co mogę sprzedać?” | compares loot with what you wear; lists what is in your bags |
| „co jest lepsze [Oily Blackmouth] czy [Linen Cloth]?”, „czy warto zachować Oily Blackmouth?” | full item facts (kind, item and required level, quality, count, vendors, drops, quests) and a side-by-side comparison; never recipes or market prices |
| „przypomnij mi za 10 minut…”, „zanotuj, że…” | reminders by time, level or place, and notes |
| „ile expa na godzinę?”, „brać need czy greed?” | session stats; loot roll advice |
| game questions („jak pokonać VanCleefa?”) | answers from the local knowledge base |
| „cicho”, „Iskra, cicho”, „milcz” | stops her answer („Iskra, cicho” works as well) |

Questions like „co jeszcze mogę zrobić?”, „co jeszcze mogę zrobić po tym queście?” go to the planner: the quest journal plus QuestieDB world data through `advisor.where_next`, so she proposes what to take next - not the diary-only `quest_info` answer. The wiki/knowledge base is only a lore fallback.

**A smarter assistant with Gemini (optional).**
- Put `GOOGLE_AI_API_KEY=...` (from Google AI Studio) in `.env` and set `"assistant": {"llm": "gemini"}` in `config.json`. Only the assistant uses Gemini; chat translation stays local.
- Her answers take 2–3 seconds instead of under a second, but they are much better in Polish.
- If Gemini fails, the local model answers.
- **Privacy:** the assistant's questions, facts from your game and names from your game (quests, items, people in chat) are then sent to Google. A free key also has a small daily limit. `"lexicon": {"embedder": "local"}` keeps the name vectors on your PC.

**Another cloud model (OpenAI-compatible).** Any OpenAI-compatible chat API can answer her instead of Gemini; translation, voices, speech recognition and the knowledge base stay local.

```json
"assistant": {"llm": "openai", "openai": {"base_url": "https://opencode.ai/inference/openai/v1", "api_key_env": "OPENCODE_API_KEY", "model": "deepseek-v4.1-flash"}}
```

- Put `OPENCODE_API_KEY=...` (the name comes from `api_key_env`) in `.env`. Only the assistant sends her questions there.
- OpenCode Zen's free chat models work only through the OpenCode client; from this app use the paid model (e.g. `deepseek-v4.1-flash`, cents per day), or point `base_url` at OpenRouter with your own model name and key.
- If the cloud fails or has no key, the local model answers. The `assistant.openai` section also takes `timeout_s` (20), `max_output_tokens` (4096) and `min_output_tokens` (2048); `timeout_s` is only the HTTP client's default - the assistant still bounds each answer with `assistant.llm_timeout_s` (8 s). `min_output_tokens` is a floor and `max_output_tokens` a ceiling for the request's `max_tokens` (0 = no ceiling): reasoning models count their thinking tokens against that cap, so a short answer request (120-400 tokens) can come back as HTTP 200 with no text at all (`finish_reason: "length"`). The floor does not force a long answer - the length still comes from the prompt. Raise the floor to 4096 if `finish=length` still shows up in the log.
- `assistant.openai.reasoning_effort` (default `low`) is sent as the top-level `reasoning_effort` field (`low` / `high` / `max` / `none`; empty sends nothing), which both OpenCode Zen and OpenRouter accept (OpenRouter treats it as shorthand for `reasoning.effort`). `low` keeps the thinking short, so answers are faster and cheaper and `finish=length` is less likely - reasoning tokens still count toward `max_tokens`. Do not write `#low` in the model name: that suffix is client-only and is never sent to the API. In Settings it is the "Reasoning effort:" dropdown in the cloud group (takes effect after a restart).
- The log line `wvt.cloud: cloud model ... failed (...) - the local model answers this one` means the cloud was asked and failed (including an empty answer with the reason, `max_tokens` and token counts), so this answer came from the local model.
- These fields (provider, base URL, model, key env, reasoning effort, router, Jev model/confidence) are also in Settings → Asystentka; provider/router changes take effect after restarting the app, and a key typed there is written to `.env`.

**Jev router (optional).** When the fast local rules do not recognise what you said, she can ask a Jev decision model which command it is, instead of treating it as ordinary chat. OpenRouter's Jev 1.14 is asked first and OpenCode Zen's Jev 1.13 answers when the first one fails or has no key.

```json
"assistant": {"router": "jev", "jev": {
  "order": ["openrouter", "opencode"],
  "openrouter": {"base_url": "https://openrouter.ai/api/alpha/decisions",
                 "api_key_env": "OPENROUTER_API_KEY", "model": "typesafe/jev-1.14"},
  "base_url": "https://opencode.ai/inference/systemone/v1/systemone",
  "api_key_env": "OPENCODE_API_KEY", "model": "jev-1.13"
}}
```

- It runs only when `parse_intent` finds nothing (a `chat` result); a command the rules already know never waits for the network.
- One request asks two questions: `is_command` (is this a command at all) and a `choice` over the command labels. A low-confidence answer or "other" leaves the normal chat/LLM path untouched, so the router can never break a command. "Not a command" is a valid answer from OpenRouter - the fallback is not asked again in that case.
- Put `OPENROUTER_API_KEY=...` (from openrouter.ai) and `OPENCODE_API_KEY=...` in `.env`. A provider whose key is missing is skipped silently; with no key at all the router is off and the local rules decide.
- OpenRouter serves Jev through its Decisions API (`https://openrouter.ai/api/alpha/decisions`), not the chat-completions endpoint; the request body is the same `{model, state, questions}`. Use `~typesafe/jev-latest` as the model to follow the newest version.
- Price is $0.042 per 1M input tokens and output is free. 1.14 against 1.13 is roughly flat for this routing task (1.14 adds image/box input and a 262k context), so the choice barely changes the answers here.
- Latency is about 230-350 ms (p50/p95 on OpenRouter) and 350-440 ms on OpenCode Zen.
- `min_confidence` (0.55) is the threshold below which the answer is ignored; `"router": "local"` (the default) turns the router off.
- The provider order, both models (`assistant.jev.model`, `assistant.jev.openrouter.model`) and both keys are also in Settings → Asystentka; a missing key skips that provider. Set `"order": ["opencode"]` to keep the old OpenCode-only behaviour and skip OpenRouter.
- **Privacy:** only the current utterance goes to OpenRouter/TypeSafe or OpenCode/TypeSafe (zero-retention) - no chat log, no quest data, and the key is never logged.
- Try it from the console (no local models are loaded):

```
.venv\Scripts\python -m wvt.cli router "weź go doproś do party"
.venv\Scripts\python -m wvt.cli cloud "Odpowiedz: działa"
```

**Game knowledge (once).** The knowledge base lets her know WoW Forever: quests, NPCs, dungeons, classes and zones.

```
.venv\Scripts\python scripts\download.py --embed-model
.venv\Scripts\python -m wvt.kb.build
```

- **What the first build does:**
  - It downloads about 600 Warcraft Wiki pages, at one page per second (about 12 minutes).
  - It downloads the WoW Forever quest data from QuestieDB.
  - It computes vectors on the CPU (about 4 minutes).
  - It translates quest texts to Polish with the local model. That takes about 40 minutes the first time; later builds translate only new quests, and `--no-translate` skips this step.
- Everything is stored in `data\kb` and works offline.
- **Your own guides:** put `.md` or `.txt` files in `data\kb\guides` and rebuild with `--offline`.

**Names from your game.** The app keeps a list of the names that matter right now: your quests, nearby NPCs, people in chat, your loot and bags.
- Speech recognition gets them as a hint.
- Misheard names are corrected by sound before translation ("Targasa" → Thargas).
- Add your own words (friends, guild, nicknames) to `data\slownik.txt`, one per line.

**Her log.** With `"privacy": {"log_assistant": true}` in `config.json`, every question, how she understood it and her answer go to `logs\assistant.jsonl`. This is off by default.

**Conversation history (in memory).** The app shows what you and Iskra said, both in an `Iskra` tab and in an always-on-top overlay that can sit over WoW.
- Both views read the same transcript kept in memory (newest first, up to 50 entries); nothing is written to disk.
- The window keeps line breaks and shows simple Markdown (bullets, bold, `#` headings as bold body text), display only; copied text is what you see.
- **Coloured names (`assistant.history_links`, on by default).** Known quests, items, NPCs, dungeons and zones in her replies show as `[Name]`, coloured by type: quest yellow (`#ffd100`), item green (`#1eff00`), NPC orange (`#ff9f43`), dungeon violet (`#c792ea`), zone blue (`#40c4ff`). Ctrl+click a quest, item or NPC link to put Questie's own details window in the WoW chat box: `/run QuestieLoader:ImportModule("QuestieSearchResults"):GetDetailFrame("<kind>",<id>)` (nothing is sent, you press Enter). A dungeon link puts the app's own `/wvt way ...` to its entrance in the chat box; zones are colour only. When the WoW window cannot be found, the command goes to the clipboard and a HUD message says so - the history window stays open (no modal box). Questie has no `/questie find item`, and `/questie search item|quest` is a TODO in Questie's source ("TODO: Implement ShowItem and ShowQuest functions"), so this details window is the closest working integration. It needs Questie with QuestieDB installed - without them the `/run` line errors in chat. Links are display only and never logged.
- **`Insert` (default) is the global history hotkey.** It shows and hides the overlay from anywhere, also in the game, as a temporary peek: it does not change the saved `assistant.history_overlay` setting. Set your own key in Settings under **Klawisz historii Iskry** (config `assistant.history_key`; `—` turns the key off).
- The **Przezroczystość okna historii** slider in Settings (30-100%) sets the overlay's window opacity live (`assistant.history_opacity`, default `1.0`).
- The overlay opens on the right of the screen a moment after start-up when `assistant.history_overlay` is on. Turn it on or off from the tray ("Historia rozmów Iskry") or with the Settings checkbox "Pokaż overlay z historią rozmów nad grą"; `assistant.history_limit` (5-30) sets how many entries both views show.
- The overlay is a normal window: you can scroll it and close it, and it does not take focus from the game when it appears. It also stays visible while the app window is hidden in the tray.
- Iskra's spontaneous comments (quest taken, level, zone, loot, bags...) also land in the history, shown in orange with a small `event: <name>` line and the entry fields `event` / `route: event` (a question answer has neither). Hide them with the "Komentarze" toggle in the overlay or the Settings checkbox "Pokaż automatyczne komentarze Iskry (np. po wzięciu questa)" (config `assistant.history_events`). Hiding them only hides those history entries; she still says the comments, just as before.
- You can type a question straight into the overlay's input and press Enter or "Wyślij" (no push-to-talk). It uses the same game context the app already has from the addon (state, quests, chat), so nothing extra is sent to the addon. The answer shows up in the transcript and the HUD; if she prepares a command, it waits in the WoW chat box and you still press Enter to send it.
- "Prompt" (in the overlay and the `Iskra` tab) shows exactly what was sent to the model last time: the system prompt, the recent turns and your question. It is read from memory only - nothing is written to the logs.
- "Bez reguł" (next to the overlay's input) arms the next question to skip the local rules and Jev, so a follow-up that would otherwise turn into `quest_worth` and the like goes straight to the conversation instead. It is one-shot: it clears itself after that one question.
- **Rules as fragments (`{...}`).** You can name the rules you want instead of hoping a phrasing matches: `{quest_available}`, `{quest_list} {quest_turnin}`, `{gold}`, `{help}`. A bare fragment runs only those rules and gives their deterministic answer. A fragment with a real question after it is hybrid: `{quest_available} co wziąć najpierw?` runs the rules first, then the LLM turns their facts into one answer and may not invent new ones. "Bez reguł" always keeps the rules-only answer and never composes. Only whitelisted rule kinds are recognised (`wvt/directives.py`); commands are excluded, and any other braces stay in the text as typed. `{` autocomplete works in both Iskra inputs (the overlay and the `Iskra` tab): type `{` and the popup lists the whitelisted rule kinds with their descriptions, e.g. `quest_list`, `quest_turnin`, `quest_available`, `quest_where`, `where_next`, `stats`, `loot`, `gold`, `help`; Tab or Enter inserts `{quest_list}`, Esc closes the list. The popup only ever offers names from the `DIRECTIVES` whitelist, so whatever it inserts is a rule the parser accepts. Arrow keys, Tab, Enter and Esc behave the same for `{` and `[`.
- **Item help (`assistant.item_help`, on by default).** `{item_about}` and `{item_compare}` answer from the world item record (`World.item_record`): kind, item level and required level, your quality and count, vendors, drops and quests. `_item_compare` puts two named items side by side, and the answer always says it has no market prices (`[A]` and `[B]` in brackets work). Item questions are routed before the ordinary rules, so "czy warto zachować Oily Blackmouth?" is about the item, while "czy warto robić ten quest?" still goes to `quest_worth` (that rule now sits lower, just before `quest_turnin`). The hybrid `{...}` answer gets an extra instruction not to add recipes, prices or uses that are not in the facts. Turn the checkbox "Rozszerzona pomoc o przedmiotach (pełne dane, porównania, bez zgadywania)" off in Settings (or set the config key to `false`) to get the old item handling and hide `item_compare`.
- **Approximate gear advisor (`assistant.gear_help`, on by default).** `{gear_next}` answers "jakie questy dają lepszy sprzęt?", "gdzie iść po upgrade itemów?" or "co robić żeby ulepszyć itemy?" with quests whose rewards could be an upgrade. It reads the rebuilt `world.all_items` index, checks the reward's class and required level against yours (`loot.WEAPONS`, `loot.SHIELD_CLASSES`, `loot.best_armor`, plus misc armor for everyone), and compares the item level with the threshold for the reward's category: weapon rewards against the average of your worn weapons, armor rewards against the average of your worn armor, with the overall average as a fallback. It lists journal quests (`oddaj` / `w dzienniku`) and, when your completed-quest list is known, available quests in your zone (`weź`) - nearest or biggest gain first. It is approximate: it does not know the reward's slot, stats or price, so it does not promise an upgrade. In the list, `poz.` is the reward's item level and `wym.` its required level. The footer lists the worn pieces the app knows, with their names and item levels, and the per-category thresholds it uses. Item level is not the same as required level. The addon now sends `TYPE_EQUIP` (type 12): a full snapshot of all 18 inventory slots with item id, item level, quality, class and subclass, but no item names - the app resolves those from QuestieDB. It is sent on login, on `PLAYER_EQUIPMENT_CHANGED` and as a 60 s heartbeat, so **the addon must be reinstalled/updated in WoW** for the app to see the full set. Without the new addon the app falls back to the old loot-only probes, where worn pieces arrive only for the slot of the looted or bag-scanned item, so the list is often short and can be stale; with only one to three known pieces in a category the answer says it is a small, possibly unreliable sample, and with none it starts from your character level. An old app ignores type 12 (the protocol version was not bumped), and an old addon simply sends none. QuestieDB has `requiredLevel = 0` for most quest gear rewards, so `wym.` is often absent: that is missing data, not "no requirement". `{gear_debug}` prints exactly what the advisor sees: whether the data is an addon snapshot or loot only, every worn piece with its slot, name, item level and class/subclass, the per-category thresholds, and every reward record with its verdict, plus the QuestieDB required-level caveat. `{gear_next_world}` widens this to the whole world: available quests with likely upgrade rewards, nearest same-map quests first and other maps after (marked "inna mapa", no distance), sorted by largest gain within each group - also approximate, same `assistant.gear_help` toggle. Turn off "Podpowiedzi questów na upgrade sprzętu (przybliżone)" in Settings (or set `assistant.gear_help` to `false`) to disable the rule and hide `gear_next`, `gear_next_world` and `gear_debug` from the `{` popup.
- **`[` autocomplete (overlay and the `Iskra` tab input).** Type `[` and a popup lists names the app already knows: quests in your journal, turn-ins, quests available to take, every QuestieDB quest name, NPCs, zones, dungeons, items, what is in your bags (the "Plecak" group) and your party members. Arrow keys pick, Tab and Enter both insert `[Name]` without changing focus to another field, Esc closes the list. The same keys work in the `{` rule popup. Sorting puts turn-ins, journal, available quests and backpack first, and the nearest place in the current zone first. A query matches the start of any word (`[storm` finds both "Stormwind" and "Messenger to Stormwind"), and typos are matched with Damerau-Levenshtein (`deadmnes` finds Deadmines). Outside the brackets no popup appears and nothing is auto-inserted; the list lives in memory only and is never logged. It can be turned off in Settings ("Podpowiedzi po [ w polach Iskry (questy, NPC, miejsca)"; config `assistant.autocomplete`).
- This needs no addon changes and no addon reinstall.

### Logs: where did the answer come from?

At the default INFO level the log says which path answered each question, without the question or answer text:

```
assistant: local=chat route=jev final=quest_info sources=QuestieDB ms=1912
Jev: label=quest_info provider=openrouter confidence=0.91 380 ms
assistant LLM: provider=openai model=deepseek-v4.1-flash variant=low fallback=False finish=stop 1840 ms
assistant LLM: provider=local model=Nemotron variant=thinking-off fallback=True fallback_from=deepseek-v4.1-flash finish=stop 640 ms
```

- `local` is what the local rules understood, `route` is `rules`, `chat`, `forced-chat` (the question skipped the rules) or `jev`, and `final` is the intent that answered.
- `fallback=True` means the cloud model failed and the local model answered; `fallback_from` names the cloud model.
- `variant` is the thinking level for Gemini/OpenAI-compatible models, or `thinking-off` for the local model.
- The text of questions and answers stays out of the log unless `privacy.log_assistant` is on; token usage is logged only at `logging.level: "DEBUG"`.

#### Full prompt history (`context_history.jsonl`)

Two separate, opt-in files answer "where did the answer come from?":

- `privacy.log_assistant: true` writes `assistant.log_file` (default `logs/assistant.jsonl`): one JSON line per question, with `heard`, `intent`, `route`, `llm` (provider/model/variant), `reply`, `sources`, `turn` and `context_refs`.
- `privacy.log_llm_context: true` writes `assistant.context_log_file` (default `logs/context_history.jsonl`): one JSON line per **assistant answer LLM call**, with `id`, `turn`, `call`, `event`, the full `messages` (system prompt, the recent turns, your question), `max_tokens`, `temperature`, `thinking`, the `llm` metadata and `error` (empty unless the call failed). The style and event examples (`STYLE_SHOTS`, `EVENT_SHOTS`) are part of the system prompt, clearly labelled as tone patterns - they are never sent as fake user/assistant turns, so they do not show up in `messages` as conversation.

Joining them:

- `context_refs` in `assistant.jsonl` are `id`s in `context_history.jsonl` - the exact prompts that produced that answer.
- `turn` groups every LLM call of one answer (a question can take more than one call, e.g. a knowledge answer); `call` numbers them 1, 2, ...
- A rule-only answer (gold, bags, invites...) has an empty `context_refs` and no context line.
- Spontaneous comments (a zone, a level up) have `event: true` and `turn: ""`.
- Context logging covers the **assistant answer model only** - not the translator and not the Jev router.

Python:

```python
import json

journal = [json.loads(l) for l in open("logs/assistant.jsonl", encoding="utf-8")]
context = {r["id"]: r for r in (json.loads(l) for l in open("logs/context_history.jsonl", encoding="utf-8"))}

for row in journal:
    print(row["heard"], "->", row["route"], row["intent"])
    for ref in row["context_refs"]:
        print("   ", context[ref]["messages"][-1]["content"], context[ref]["llm"])
```

PowerShell, last prompt only:

```powershell
Get-Content logs\context_history.jsonl -Tail 1 | ConvertFrom-Json | Select-Object -ExpandProperty messages
```

Both files are private and off by default. They grow until you delete them - turn them on to debug a wrong answer, then turn them off again. Changes to `privacy.log_assistant`, `privacy.log_llm_context` or the log file paths apply after restarting the app, because the assistant opens those files at startup.

## Configuration

`config.json` is created on the first start. Most settings are also in the app window.
- `push_to_talk`: keys and channels.
- `audio`: microphone and auto gain.
- `voice`: volumes, distance curve, daily character budget, `/say` safety valve.
- `assistant`: name, voice, LLM, which comments she makes.
- `lexicon`: names from your game.
- `privacy`: what is logged.

Every key is described (in Polish) in [docs/SZCZEGOLY.md](docs/SZCZEGOLY.md).

## Updating

```
git pull
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The setup script only installs what is new and copies the current addon into the game. Then restart the app and type `/reload` in the game.

## Troubleshooting

- **No hover translations, and `/wvt test` shows nothing.**
  - Use *Windowed (Fullscreen)*, and keep WoW the active window.
  - Make sure the app has finished loading: `logs\wvt.log` says when the models are ready.
  - Keep the top-left corner of the game free of overlays: FPS counters, Discord or Steam overlays.
  - **HDR on?** Tone mapping clips the bright cell levels, so the grid is visible but the app cannot calibrate it. Run `/wvt hdr on`: the cells get darker (0/15/30/45%) and the app calibrates itself from what it captures.
  - If that still fails, use SDR for the game, or turn off Auto HDR for it: HDR GDI capture clips colors and the grid may not be readable at all.
- **Setup didn't find WoW.** Run it with `-Wow "D:\Games\World of Warcraft"`, the folder that contains `_forever_`.
- **Speech isn't recognised.** In the app window, go to *Settings → Microphone*, pick the right microphone and run the test.
- **The microphone is quiet.**
  - The app boosts quiet speech by up to +30 dB before recognition. The microphone test shows the speech level; −30 to −15 dBFS is good.
  - It is better to raise the level at the source: Windows sound settings, *Recording → Levels*.
- **Voices are silent.**
  - Check the app's **Voices** tab: the key, the number of voices and the daily limit.
  - Check that **Say** or **Party** is on in the game (`/wvt voice`).
  - After an addon update, type `/reload`.
- **Starting is slow.** The models are large. From a slow or USB 2.0 drive, starting takes a few minutes. Leave the app running in the background.
- **Diagnostics.** Run `.venv\Scripts\python -m wvt.cli check`. Logs are in the `logs` folder.

## Safety and privacy

- **Nothing touches the game.** The app doesn't read game memory, injects nothing into the game and never sends chat messages for you. It only pastes text into the chat box, and you press Enter.
- **The addon** uses only the official addon API. It draws the signal grid in the corner of the screen.
- **Recordings and chat.** Recordings are not saved. Chat content is not saved, unless you turn on the assistant's log.
- **Your keys** live only in `.env`, which is git-ignored.
- **ElevenLabs** receives only the text of the lines from channels you switched on.
- **Google** receives data only when you choose Gemini for the assistant (see above).
- **Your responsibility.** Use the app at your own risk, and check Blizzard's rules on third-party software.

## Tests

```
.venv\Scripts\python -m pytest
```

The addon's Lua code is tested with `lupa` (Lua 5.1, like WoW). `tests\test_capture_integration.py` draws on the screen, so it fails while the app's overlay is running.

## More

- [docs/VOICES.md](docs/VOICES.md): creating the character voices (ElevenLabs).
- [docs/SZCZEGOLY.md](docs/SZCZEGOLY.md): how it works, measurements, model choice and all settings (in Polish).

## License

The code is MIT; see [LICENSE](LICENSE).

The models are downloaded by the setup script. They are not part of this repository and have their own licenses:
- OpenAI Whisper large-v3-turbo: MIT;
- NVIDIA Parakeet TDT 0.6B v3: CC-BY-4.0;
- NVIDIA Nemotron 3.5 Bolt E4B: OpenMDW-1.1;
- multilingual-e5-small: MIT;
- WeSpeaker ResNet34: Apache-2.0;
- llama.cpp: MIT.

ElevenLabs voices are covered by the terms of your ElevenLabs plan.

The knowledge base is built on your PC and is not part of the repository:
- Warcraft Wiki texts are CC BY-SA 4.0.
- QuestieDB data has no single clear license, so use it only privately.
