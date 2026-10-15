# NeuroBridge

A brain-translation **training system** for one specific piece of hardware: the
NingNao / SereniBrain **TH21A** BLE headband — a single dry frontal electrode at
250 Hz that you already reverse-engineered and can stream from Linux
(`../ble/README.md`).

NeuroBridge gives that channel a job: **64 stimulus protocols**, a recording
runner with a real marker timeline, an analysis pipeline that trains and
honestly evaluates decoders, and a bridge that feeds the decoded state to your
AI agent.

---

## Read this first: what one frontal electrode can and cannot do

This is a single dry electrode on the forehead. There is no spatial
information, no occipital coverage, no calibrated amplifier, and the signal
drifts as the electrodes dry out. Capability is therefore bounded, and the
whole system is built to be honest about it:

| This system can train and verify | This system will never do |
|---|---|
| Alpha reactivity (eyes open vs closed) | Read words or sentences |
| Arousal / alertness / drowsiness level | Decode imagined images |
| Gross attention and cognitive-load proxies (theta/beta) | Report what you are looking at |
| Blink and jaw-clench **commands** (large, reliable artefacts at Fpz) | Lateralised motor imagery (needs C3/C4) |
| Frontal P300 / MMN dichotomies with long averaging | Fast multi-class mental typing |
| Affective arousal (weakly), image/audio category effects | Valence via frontal asymmetry (needs 2+ channels) |

Every one of the 64 tests carries an explicit `feasibility` band —
`established`, `promising`, `exploratory`, or `control` — and the reports hold
each one to a gate appropriate to that band. `exploratory` tests are worth
recording and are *expected* to fail sometimes; `control` tests are designed to
fail, and are how the pipeline proves it is not fooling itself.

If you want the shortcut version of the honest claim: **this is a state sensor
and a two-or-three command switch, not a mind reader.** The value is that the
state it reports is measured, gated, and auditable rather than vibes.

---

## Install into the system menu

```bash
./scripts/install_desktop.sh              # per-user, no root needed
./scripts/install_desktop.sh --system     # all users (needs sudo)
./scripts/install_desktop.sh --uninstall  # remove the entries
```

That installs three entries and their icons into `~/.local/share`:

| Menu entry | What it opens |
|---|---|
| **NeuroBridge** | the Tkinter control panel — check everything, plan sittings, record a session, simulate, evaluate, freeze splits, record/analyse voice, open the docs |
| **NeuroBridge Agent Bridge** | a terminal running `nb serve`, for when your AI agent needs the live state endpoint |
| **NeuroBridge Monitor** | the standalone monitor: live trace, CSV recording, and replay of previous sessions |

It appears under **Science** in the MATE menu, or just type "NeuroBridge" in the
menu search. If it does not show up immediately the panel menu is caching —
`mate-panel --replace &`, or log out and back in.

The control panel is a *front-end*, not a second implementation: every button
shells out to `python -m neurobridge <args>`, so what you click is exactly what
`./bin/nb` runs, with the output streaming into the console pane. Experiments
still open their own pygame window, because stimulus timing must not go through
a toolkit event loop.

```bash
./bin/nb gui        # the same control panel, from the command line
./bin/nb-gui        # what the menu entry actually executes
```

The project is also importable from anywhere (`nb` is on the venv's `PATH`), so
`<project>/.venv/bin/nb` works from any directory.

## Install / run

The project venv is already built with every dependency (numpy, scipy,
scikit-learn, MNE, torch, braindecode, pygame, sounddevice, pylsl, bleak,
FastAPI). Use the launcher:

```bash
cd /path/to/neurobridge
./bin/nb doctor            # environment, headband, audio, credentials
./bin/nb catalog list      # all 64 tests
```

`doctor` is the first thing to run on any new machine: it checks Python,
optional packages, the data tree, catalog validity, the display and audio
device, whether the headband is advertising (`--scan`), your Freesound token,
and Openverse reachability.

### First session (no hardware needed)

```bash
./bin/nb simulate T301      # synthetic session + full analysis, ~20 s
```

This is not a toy: it generates a session through the *same* store format the
headband writes, then runs the real pipeline over it. Because the synthetic
source injects a known P300 with known amplitude and jitter, a passing result
proves the timeline, marker stream, epoching and decoder all agree. `--effect-size 0`
removes the injected effect and must therefore land at chance — that is the
false-positive check.

### First session (hardware)

```bash
./bin/nb run T002 T003 -y   # ~8 min: resting baselines + alpha reactivity
```

Put the headband on first: switch on the nose bridge, band just above the
eyebrows, all electrodes flat on skin with **no hair trapped**. Wipe the
electrodes with saline or alcohol if contact is poor — see `docs/HARDWARE.md`.

---

## The 64 tests

Eight tiers of eight tests. The full machine-readable catalog is in
`neurobridge/catalog/tests/` (one YAML per test); `./bin/nb catalog show T301`
renders any of them, and `./bin/nb catalog stats` summarises the set.
**[docs/CURRICULUM.md](docs/CURRICULUM.md)** is all 64 in one readable document,
regenerated with `nb catalog docs`.

| Tier | Theme | What it trains | Typical use by the agent |
|---|---|---|---|
| 0 | Commissioning & signal integrity | contact quality, latency calibration, artefact library, null floor | none — protects every other test |
| 1 | Arousal, alertness & fatigue | alert/drowsy, vigilance decrement, arousal from audio | `state` |
| 2 | Attention & cognitive load | n-back, arithmetic, Stroop, speech-in-noise load | `state` |
| 3 | Auditory ERPs | P300 oddball, MMN, speech syllables, own name | none — validates the chain |
| 4 | Visual & semantic ERPs | image oddball, RSVP, N400, lexical decision | none |
| 5 | Reading & language | self-paced reading, aloud vs silent, word frequency, readability | `state` |
| 6 | Audio listening & music | music/speech/noise, tempo, consonance, soundscapes | `state` |
| 7 | Imagery, control & closed loop | blink/clench switches, imagery, agent handshake, neurofeedback | `command` / `both` |

Each spec fixes: the design (trials, blocks, SOA, jitter, duration), the exact
marker codes, the analysis recipe (epoch window, bandpass, features, decoder,
cross-validation), the acceptance gates, the honest failure modes, and how the
agent is allowed to use the result. Nothing about the analysis is chosen after
seeing the data.

Text reading, image viewing and audio listening are all covered — images come
from the **Openverse API** (`https://api.openverse.org/v1/`), audio from
**Freesound** (`https://freesound.org/apiv2/`), and text from a local
public-domain text bank.

### A 45-minute session plan

A sitting is ~45 minutes with the headband on. Suggested curriculum order:

1. **Day 1** — `T001 T002 T003 T007` (commissioning, baselines, artefact
   library). Do not skip: T003 must show alpha rising on eye closure or the
   electrode contact is not good enough to record anything else.
2. **Day 2** — `T004 T005 T006` (latency calibration → sets
   `output_latency_offset_s` in `neurobridge.yaml`).
3. **Day 3–4** — `T301 T304` (auditory P300 and MMN: the cleanest ERP tests).
4. **Day 5+** — one state test per session from tiers 1–2 (`T101 T201 T104`…),
   then tier 6 listening tests, then the tier 7 control tests once blink/clench
   detection is verified on your own artefacts.
5. `T008` (sham) at least twice: it must stay at chance.

---

## Companion protocol (imported)

A second, independently written 64-test specification was imported into this
repository and reconciled with the catalog:

| | |
|---|---|
| Document | [docs/EEG_AGENT_TRAINING_PROTOCOL.md](docs/EEG_AGENT_TRAINING_PROTOCOL.md) — verbatim, with a sha256 provenance sidecar |
| 64 cards, machine-readable | [neurobridge/catalog/protocol_codex.yaml](neurobridge/catalog/protocol_codex.yaml) (modules A–H, ids `T01`–`T64`) |
| Media ledger | [neurobridge/catalog/media_ledger.yaml](neurobridge/catalog/media_ledger.yaml) — 20 candidate Openverse/Freesound assets with licences, attributions, Openverse ids and media URLs |
| Crosswalk | [docs/PROTOCOL_MAPPING.md](docs/PROTOCOL_MAPPING.md) — every card mapped to a NeuroBridge test, with the gaps named |

It is a **content and epistemics battery**; NeuroBridge is the acquisition and
analysis machinery. The crosswalk is explicit about the one thing that must not
be lost: those cards are 64 *test types*, one 8–13 s exposure each, and one
exposure per concept cannot train or validate a classifier. The concepts
(water / fire / dog / piano, across text, image and sound) are the right
stimuli — hosted inside NeuroBridge's paradigms, at 40+ trials per class.

Its label policy, its "no authorization from EEG" rule, its timing-honesty
requirement and its frozen development/locked-evaluation split are all enforced
in code (see the bridge section above and `nb split`).

## Voice channel (microphone)

Reading is mostly behaviour, and one frontal dry electrode cannot see behaviour.
With `--mic` the session also records the microphone, on the same
`time.perf_counter()` clock as the EEG packets and the markers:

```bash
./bin/nb run T502 --mic          # reading aloud vs silent, with voice captured
./bin/nb voice --session latest  # per-trial speech metrics
./bin/nb voice --record 60       # standalone sample, no headband needed
./bin/nb doctor --mic            # list input devices + a 2 s level probe
```

Per trial you get **cue-to-speech onset latency, speech duration, speech rate
(envelope-peak proxy), pause/restart count, voiced fraction and level** — so
"did I actually read that aloud, and how did I process it" becomes a
measurement rather than a self-report. `speech_mask()` is a deliberately simple
energy+ZCR detector; it answers "did speech start, and when" well (verified
against synthetic ground truth: 500 ms onset → 500.0 ms, 2000 ms speech →
2000.0 ms, a 300 ms gap → exactly one pause).

Four things worth being precise about:

* **Onset latency is not a cortical latency.** It is cue-to-voiced-audio and
  includes speaker, buffer and microphone latency. Run
  `voice.measure_roundtrip()` — play a tone, find it in the recording — to
  measure this machine's round trip before interpreting absolute values.
* **Speech rate is a proxy.** 2–8 amplitude peaks per second track syllable
  nuclei; it is not a word rate. For word-level timing, run a real aligner over
  `audio.wav` and join on the timestamps.
* **Voice metrics never enter the EEG feature matrix.** A classifier trained to
  predict "spoke" from audio-correlated features has learned the microphone.
  Use voice as the *target* ("does frontal theta predict onset latency?") or as
  a confound check.
* **The confound it exposes is the point.** Reading aloud moves the jaw, and jaw
  EMG is large at Fpz. `T502` declares this: if the audio envelope and gamma
  power track each other per trial, that classifier learned the jaw, not cortex.

**Privacy.** Microphone capture is opt-in per session (`--mic`), written to
`audio.wav` inside that session directory, and never uploaded anywhere.
Consent, device name and timestamp are recorded in the session metadata. The
`doctor --mic` probe discards its 2 s sample by default; nothing else records
audio unless you ask it to.

## During a session

* **Fullscreen by default.** Stimuli cover the whole screen on the real desktop
  resolution, with the cursor hidden, so the wallpaper, panels and whatever else
  is on screen cannot distract you mid-block. The dark NeuroBridge screen stays
  up for the *entire* session — including instructions, cues and breaks — so the
  desktop is never visible between phases. The screensaver and display blanking
  are inhibited for the duration and released afterwards. Opt out with
  `nb run --windowed`.
* **Nothing advances by itself.** In the default `--advance key` mode every
  instruction, cue and break waits for **SPACE**, with no countdown pushing you
  on. Use `--advance timer` for rehearsal, and note that a run with no display
  and no keyboard (automation, CI) falls back to the timer automatically and
  says so — otherwise it would wait forever.
* Keyboard: **SPACE/RETURN** advances and records a response, **q** or **ESC**
  aborts and still writes a valid session.

## Reporting a problem (logs)

Everything the control panel runs is appended to a dated log automatically, so a
problem can be reported even if you never pressed anything:

| Where | What |
|---|---|
| `data/logs/commands-YYYYMMDD.log` | every command, with a header (settings, device, backends, display) and per-line timestamps |
| `data/sessions/<id>/run.log` | what a session itself printed, timestamped with elapsed seconds |
| `data/logs/diagnostic-*.txt` | one shareable bundle: versions, settings with **credentials withheld**, catalog state, session summaries, doctor output, recent logs and any recent tracebacks |

Three ways to produce them:

```bash
./bin/nb diagnose --note "the session stopped after block 2"   # one file to attach
./bin/nb diagnose --scan --mic --print                        # include a BLE scan + mic probe
```

or, in the control panel, **Logs ▸ Collect diagnostics…** (writes the file *and*
puts its path on the clipboard), **Save console log…**, **Copy console log**, and
**Copy last session's run.log**. Diagnostics are strictly read-only: they add a
report file and never modify or delete session data.

## How a session works

```
headband ──BLE──► device/th21a.py ──packets+perf_counter──► store.SessionWriter
                                        │
                    runner/session.py ──┼── markers (SessionMarkerSink, optional LSL,
                                        │            and FakeSource.notify_marker)
                                        │
              present_visual / present_audio ── stimulus onset timestamps
                                        ▼
                    data/sessions/<id>/{meta.json, eeg.f32.npy, time.f64.npy,
                                        events.jsonl, notes.md, summary.json}
```

The timing model is the part that matters:

* **one clock** — `time.perf_counter()` — stamps both stimulus onsets and BLE
  packet arrivals, in the same process;
* BLE delivers 10 samples per notification, bursty, so per-sample times are
  estimated backwards from the arrival time at the nominal 250 Hz;
* all waiting in the runner goes through a ≤5 ms poll loop, so acquisition
  never stalls during a break and the display never freezes;
* the residue — audio output latency, display flip latency, BLE transport
  delay — is **measured**, not guessed, by tests `T004`/`T005`, and applied as
  `output_latency_offset_s`;
* ERP protocols always use jittered inter-trial intervals, so the evoked
  response cannot hide behind ongoing alpha.

## The analysis pipeline

`preprocess` (notch 50/60 Hz, zero-phase bandpass, drift and artefact
detection) → `epochs` (stimulus-locked, or fixed windows for sustained-state
tests, grouped by block) → `features` (band power and ratios, ERP window
amplitudes, Hjorth, spectral entropy, 1/f slope, covariance) → `decoders`
(LDA, logistic regression, SVM, a dependency-free Riemannian tangent-space
classifier, and a 1-channel EEGNet) → `evaluate` (leave-one-block-out
cross-validation, permutation test, Wolpaw information transfer rate, and the
spec's acceptance gates) → `report` (figures + a markdown verdict per test and
a 64-row curriculum table).

Two design choices are worth knowing:

* **Grouped cross-validation by block.** Shuffling trials would let the
  classifier memorise slow drifts that happen to correlate with condition. All
  reported accuracy is on held-out blocks.
* **Relative power only.** The device is uncalibrated, so absolute amplitudes
  are meaningless; every feature is either relative, log-ratio, or z-scored
  against a personal baseline captured at the start of the session.

## The brain-to-agent bridge

```bash
./bin/nb serve                 # live headband
./bin/nb serve --device fake   # no hardware: synthetic, for wiring up the agent
```

Then point your agent at `http://127.0.0.1:8731/`:

| Endpoint | Purpose |
|---|---|
| `GET /schema` | self-describing contract: every field, range, meaning, and pasteable tool definitions |
| `GET /state` | current `attention`, `arousal`, `load`, `fatigue`, `relaxation`, `confidence`, `artifact_level`, `contact` |
| `GET /state/stream` | SSE, 2 Hz, with `seq` and `t_mono` so drops are detectable |
| `WS /ws` | same payloads over WebSocket, bidirectional |
| `POST /mark` | the agent writes its own markers into the EEG timeline |
| `GET /events` | merged marker + detected-intent events |
| `GET /report/{test_id}` | latest evaluation for a test |

Until a decoder is trained, every field is a **baseline-relative heuristic**
and the payload says so in `model_info`. `valence` is always `null` on one
channel and the schema explains why rather than inventing a number. Intents
(`blink`, `clench`) are the only commands the hardware supports reliably.
`neurobridge/agent/protocol.md` is written for an LLM consumer, including the
"what this cannot do" section.

### The client half: `nb link`

`nb serve` publishes state; `nb link` consumes it. It exists because an agent
thinks in discrete turns, so "what is the value *now*" is usually the wrong
question — "what happened *while we were talking*" is the interesting one.

```bash
./bin/nb-link status                 # up? real person? stream advancing?
./bin/nb-link record                 # log state to data/logs/agent-link-<date>.jsonl
./bin/nb-link report --last 10m      # trends for that window, plus its caveats
./bin/nb-link --json report --last 10m   # the same, machine-readable
./bin/nb-link calibrate              # establish the baseline the fields need
```

### Establishing the baseline

Every 0..1 field is relative to a personal baseline, and until one exists they
are all pinned at `0.5` with derivation `uncalibrated_neutral` — which carries no
information about anyone. Nothing captures that baseline automatically, because
the moment it is captured defines the reference for the whole session: an
automatic capture would freeze whatever you happened to be doing at the time.

So seat the band, check the mount quality with `./bin/nb-link now` (read
`model_info.quality.messages` — it names drift, mains hum, saturation and motion),
and when the capture looks clean:

```bash
./bin/nb-link calibrate              # ~60 s of session start becomes the baseline
```

It refuses to do so when the source is not a live headband, because a baseline
from generated data would be relative to nothing about you. `--force` overrides
that for development only. The baseline is persisted per session
(`data/models/baseline_<session>.json`), so a bridge restart mid-session adopts
it instead of silently reverting the fields to `0.5`.

Two properties are enforced rather than left to the caller:

* **Provenance comes from `/health`, not `/state`.** The bridge can serve a live
  person, a replayed recording or a synthetic generator, and `source` in
  `/state` names the *requested* device alias (`auto`), not what is really
  producing samples. The link reads the resolved device from `/health`
  (`device` vs `requested_device`), so a generated number can never be rendered
  as someone's brain state.
* **A field with no baseline is not a measurement.** Until a personal baseline
  exists every field is pinned at `0.5` with derivation `uncalibrated_neutral`;
  the link prints `-` for those instead of `0.50`, and
  `nb link report` computes the specific caveats for the window it summarised
  (synthetic source, uninformativeness, mean confidence below the action gate,
  stream gaps) instead of leaving them to be remembered.

Recorded logs begin with a self-describing header carrying the provenance, so a
log read hours later by a different agent still says whether it ever described a
person.

Three rules from the imported companion protocol are enforced in the payload
(see [PROTOCOL_MAPPING.md](docs/PROTOCOL_MAPPING.md)):

* **A candidate, never a class.** `GET /candidate` returns
  `candidate_label: "uncertain"` whenever quality, confidence or a validated
  decision rule is missing, with `probabilities_calibrated: false` unless a real
  calibration step ran. It never falls back to the most likely class.
* **Three separate labels.** Events carry `task_label` (what was presented or
  asked for), `reported_label` (what you actually reported, including
  skipped/failed/unknown) and `predicted_label` (what the model estimated). A
  picture of a forest is a forest stimulus, not evidence that you felt calm.
* **An EEG estimate is not authorization.** A predicted `yes` is not consent and
  must not trigger any consequential or external action. The only thing that may
  follow from it is a reversible comfort action such as pausing stimuli, with a
  manual stop always available.

## The monitor

`nb monitor` (or `./bin/nb-monitor`) is a standalone window for recording and
watching a session, separate from both the control panel and `nb serve`. It shows
a scrolling trace, the band-power split, the five state fields, and the signal
quality — and writes what it sees to disk.

The **view** selector on the toolbar sets how much of the trace is on screen:
**1, 5, 10, 15, 30 or 60 minutes** (default 1). History is always retained up to
the full hour, so widening the window uncovers the past rather than showing an
empty chart; long views redraw more slowly because an hour at 250 Hz is ~900k
points. `--view-minutes` sets the initial length and `--window` overrides it with
an exact number of seconds for a close look at a few seconds.

```bash
./bin/nb-monitor                          # own the headband: record + display
./bin/nb-monitor --view-minutes 30        # start on a 30-minute view
./bin/nb-monitor --window 6               # a 6-second view instead
./bin/nb-monitor --replay data/sessions/s-20261014-230857-01b2
./bin/nb-monitor --url http://127.0.0.1:8731   # attach to a running bridge
./bin/nb-monitor --headless --device th21a --seconds 300   # no display
```

It is a separate process on purpose, so it keeps recording if the control panel
closes (and vice versa), and the control panel's **Monitor** buttons just spawn it
detached.

**Three sources, because only one process can hold the BLE link:**

| mode | when to use it | what it can record |
|---|---|---|
| `--device` | nothing else is using the headband | raw EEG + state |
| `--url` | `nb serve` is already running | raw EEG (via `/samples/stream`) + state |
| `--replay` | looking at the past | nothing — playback only |

**What it writes** — into `data/sessions/<session-id>/`:

* `eeg.csv` — full rate, one row per sample: `sample,t_mono,t_rel_s,value`
* `state.csv` — one row per state update (~2 Hz) with the five fields,
  confidence, artifact level, contact and the source on every row
* `events.csv` — markers
* `monitor.json` — the provenance block, so the CSVs are self-describing
* the **native session format** as well, so `nb eval`, `nb train` and `nb report`
  can use a monitor recording exactly like one from `nb run`

CSV values are **raw uncalibrated ADC counts**, not microvolts, and `t_mono` is
`time.perf_counter` — both are stated in `monitor.json` rather than left to be
guessed.

## Stimuli

* **Images** — Openverse, cache-first, **requesting** commercially-usable licences
  (CC0/BY/BY-SA), downscaled to ≤1024 px, with every item's licence and
  attribution recorded in `data/stimuli/ATTRIBUTION.md`. Works anonymously; add
  `NB_OPENVERSE_CLIENT_ID`/`_SECRET` for a higher rate limit.

  **Read the attributions before you reuse the media.** The licence is requested
  per query, but the cache key does not include it (see the known limitation
  below), so a cache hit can return something fetched under broader terms. In the
  cache this project was developed against, 56 of 87 attributed works are
  **NonCommercial** (`by-nc-nd`, `by-nc-sa`, `by-nc`) and 4 are NoDerivatives.
  Those terms are **not** compatible with the commercial use the GPL-3.0 grants
  for this *code*, and the media is not distributed with it. For a clean cache,
  delete `data/cache/stimuli` and fetch with a strict licence filter, or replace
  downloaded images with the built-in procedural ones (`data/stimuli` is
  git-ignored, so nothing here ships the media itself).
* **Audio** — Freesound, cache-first. Freesound requires a token even to
  search, so set `NB_FREESOUND_TOKEN` (or `freesound_token:` in
  `neurobridge.yaml`) to use real recordings.
* **Offline fallbacks** — with no token or no network, audio is *synthesised*
  (tones, click trains, formant-synthesised syllables such as /ba/ vs /da/,
  metronome loops, pseudo-speech and its time reversal) and images are drawn
  procedurally. The fallbacks keep every test runnable and each substitution is
  reported in the `warnings` of the resolved stimulus set. They are not a
  substitute for real photographs or recordings — the report says which was
  used.
* **Text** — a local bank of self-authored and public-domain material
  (`data/stimuli/text/textbank.json`), including word-frequency lists,
  pseudowords, garden-path sentences, N400 frames, readability-graded passages,
  Stroop pairs and digits for the SART.

Offline, all 64 specs resolve to **165 concrete items** with no token, no
network and no display — verified by an independent sweep, not by assertion.
Real downloads were also verified against the live APIs: Openverse images come
back PIL-validated, downscaled and licence-attributed.

**Known limitation** (flagged by the stimuli workstream and worth fixing before
serious use): the Openverse cache key does not include the licence filter, so a
cache hit can return an image originally fetched under a broader CC licence than
a later spec requested. Every cached item is CC-licensed with its attribution
recorded, so the practical risk is small — but if you need strict per-spec
licence filtering, clear `data/cache/stimuli` (or add the licence to the cache
key in `stimuli/openverse.py`).

## Commands

```
nb doctor                     environment / hardware / credential checks
nb catalog list|show|validate|stats
nb fetch T601 --limit 4       resolve (download or generate) stimuli only
nb run T002 T003              live session (add --device fake for synthetic)
nb run T301 --dry-run         timing rehearsal: markers and instructions, no samples
nb simulate T301              offline synthetic session + full analysis
nb simulate T301 --effect-size 0     null-model check (must stay at chance)
nb train T201 --all-sessions  train and save a decoder
nb eval T301 --session latest evaluate one test against its gates
nb report --curriculum        64-test progress table
nb plan --remaining           pack what is left into 45-minute sittings
nb progress                   per-test status: recorded, trained, pass/fail
nb catalog audit              structural checks the schema cannot make
nb split create --spec T201   freeze a development / locked-evaluation split
nb train T201 --split default fit on the development side only
nb eval T201 --locked         held-out evaluation on the locked sessions
nb run T002 T003              live session (fullscreen; SPACE advances)
nb run T301 --advance timer   auto-advance instead of waiting for SPACE
nb run T003 --windowed        run the stimuli in a window
nb run T502 --mic             record the microphone alongside the EEG
nb screen-test                show the session surface and report its coverage
nb diagnose --note "…"        write one shareable diagnostic bundle
nb voice --session latest     per-trial speech metrics (onset, duration, pauses)
nb voice --record 60          standalone voice sample, no headband needed
nb sessions list|show|export
nb state --seconds 45         one-shot live state estimate
nb serve                      agent bridge
nb link status                is the bridge up, and is it a real person?
nb link now                   one honest snapshot, with provenance
nb link watch --seconds 60s   follow the live stream in the terminal
nb link record                log the stream to JSONL while a conversation runs
nb link report --last 10m     summarise that window, with its caveats
nb link calibrate             capture the personal baseline the fields are relative to
nb link mark agent_prompt_shown   write an agent marker into the EEG timeline
nb monitor                    standalone window: live trace + CSV recording
nb monitor --replay <folder>  play back a recorded session
nb monitor --url <bridge>     attach to a running `nb serve` instead of the device
nb monitor --headless --device th21a --seconds 60    record with no display (SSH)
```

### Locked evaluation (do this or the numbers are decoration)

`nb split create --spec T201` writes a frozen manifest that makes the newest
sessions **locked**, and `nb train` then refuses to fit on them. `nb eval
--locked` trains on the development sessions and scores the locked ones. Until
you have done this, every accuracy figure is a within-session estimate from a
single seating, which is the weakest kind of evidence there is — the `train`
command says so out loud when it only has one session.

### Real recordings you already have

`../ble/session_packets.txt` (your eyes-open/eyes-closed capture) can be turned
into a NeuroBridge session without re-running anything:

```bash
.venv/bin/python scripts/import_ble_packets.py ../ble/session_packets.txt --test-id T003
```

This decodes the packets with NeuroBridge's parser — verified byte-identical to
your original decoder over 15,060 real samples — and writes a session with the
`eyes_open` / `eyes_closed` class markers, so `nb eval T003` works on real data
immediately. It also independently reproduces your alpha result: **2.76 %
eyes-open → 6.18 % eyes-closed, ×2.24**, versus the ×1.92 you measured. The
import is tagged `timing_quality: coarse`, because synthesised sample times are
fine for band power and useless for ERP latencies.

### On runtime honesty

A full evaluation includes the permutation test each spec pre-registers (200
label permutations inside the grouping unit), so expect roughly **30–60 s per
test** and **30–45 minutes for all 64** on this machine. That cost is the
analysis being honest rather than a bug: the p-value comes from actually
shuffling labels within blocks, not from a normal approximation. `nb eval` on a
single test is the normal workflow — `scripts/verify_all.py` is the sweep.

`nb catalog audit` compares each spec's *declared* `duration_min` against the
duration you actually get by building its timeline from its own trials, SOA,
jitter, block and condition keys. Across the 64 tests the two disagree in both
directions for 30-odd specs — some declared values omit jitter and breaks, and
some mechanical estimates miss cue/response windows that no formula knows about.
The audit prints every disagreement, and `nb plan` books the **larger** of the
two so a sitting is never under-booked. Planning total: **~768 minutes, i.e.
about 17 sittings at 45 minutes**, plus repeats of the commissioning tests.

The audit also verifies what the schema cannot: that every spec's design
actually builds a timeline, that every feature string is in the implemented
vocabulary, that acceptance gates sit inside the band implied by the spec's
`feasibility`, and that no single-class test carries an accuracy gate it cannot
compute.

## Layout

```
neurobridge/
  bin/nb                     launcher (uses .venv)
  bin/nb-gui                 control panel (the system-menu entry runs this)
  bin/nb-serve               agent bridge with a banner (second menu entry)
  assets/                    procedurally generated icons
  neurobridge/
    config.py store.py cli.py simulate.py splits.py voice.py gui.py
    device/    th21a.py (live BLE), fake.py (synthetic + null model), quality.py
    catalog/   schema.py, tests/T0xx_*.yaml … (64 specs),
               protocol_codex.yaml + media_ledger.yaml (imported companion)
    stimuli/   openverse.py freesound.py synth.py textbank.py builtin.py builder.py cache.py
    runner/    session.py timeline.py present_visual.py present_audio.py
    pipeline/  preprocess.py epochs.py features.py decoders.py evaluate.py report.py
    agent/     state.py bridge.py link.py protocol.md
    monitor/   sources.py recorder.py app.py      standalone monitor
  data/        sessions/ models/ reports/ splits/ stimuli/ cache/
  docs/        CURRICULUM.md (all 64 in one document)
               VERIFICATION.md (every claim with its evidence and the gaps)
               EEG_AGENT_TRAINING_PROTOCOL.md + PROTOCOL_MAPPING.md (imported)
               CATALOG_SPEC.md INTERFACES.md HARDWARE.md METHOD.md
  scripts/     install_desktop.sh (system-menu entries)
               make_icon.py (draws the icons)
               import_ble_packets.py (your raw captures -> sessions)
               import_protocol.py (re-import/refresh the companion document)
               verify_all.py (all 64 tests end-to-end on synthetic EEG)
               first_session.sh (guided first hardware sitting)
  tests/       test_smoke.py
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `headband not advertising` | long-press power ~3 s; `btmon`/`bluetoothctl` to confirm it is paired |
| Delta > 90 %, `SATURATION` in the quality report | electrodes are off the head or dry — clean with saline/alcohol and reseat; the pipeline refuses to report a state from a `POOR` capture |
| No alpha increase with eyes closed | contact problem, or you are too alert — reseat and re-run `T003` before trusting any other test |
| Accuracy at chance on everything | check `T301` synthetic first (`nb simulate T301`); if that passes, the problem is contact, not code |
| Marker jitter warnings | close other apps using Bluetooth/Wi-Fi; the runner prints mean/σ/max jitter per session |
| Freesound returns 401 | the API needs a token for search too — set `NB_FREESOUND_TOKEN` |

## Honest limitations, in one place

1. **One channel means no topography.** Anything that depends on *where* a
   signal is (N170, SSVEP, motor imagery, lateralised readiness potentials) is
   either unavailable or reduced to a weak frontal proxy, and the affected
   specs say so.
2. **Absolute amplitude is meaningless.** No calibration data exists for this
   device, so nothing here reports microvolts.
3. **Dry electrodes drift and saturate.** Quality is assessed per session and
   `POOR` captures are excluded from training rather than quietly averaged in.
4. **A passing gate is not proof of a brain signal.** It means the decoder beat
   chance on held-out blocks with the pre-registered features. The `control`
   tests exist to keep that claim honest, and `nb simulate --effect-size 0`
   lets you re-verify the pipeline's floor at any time.
5. **Nothing here is a medical device**, and none of it should be used for
   diagnosis, treatment, or any decision that matters.

## Licence

**GNU General Public License v3.0 or later** (GPL-3.0-or-later) — see
[LICENSE](LICENSE). You are free to use, study, share and modify this project,
including commercially; if you distribute a modified version, it must stay under
the same licence and you must make your source available. There is no warranty.

    Copyright (C) 2026 the NeuroBridge contributors

    This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
    details.

### Third-party material in this repository

* `docs/EEG_AGENT_TRAINING_PROTOCOL.md` is a **verbatim copy** of an upstream
  protocol document (97,280 bytes, sha256 `7625b961…`), imported by
  `scripts/import_protocol.py` and recorded in
  `docs/EEG_AGENT_TRAINING_PROTOCOL.provenance.json`. It is included as a
  specification the code was written against; copyright in it rests with its
  author, and the GPL above covers this project's code, not that document.
  Remove it, or replace it with your own, if you redistribute under other terms.
* `neurobridge/catalog/protocol_codex.yaml` and `media_ledger.yaml` are derived
  from that document by the same script.
* **No recorded data is published.** `data/` is git-ignored: recordings are
  biometric and the stimuli are reproducible from the catalog. The one exception
  is `data/stimuli/ATTRIBUTION.md`, which is committed so that media licences
  travel with the code that fetches the media.

### Nothing here is a medical device

This is research tooling for one consumer dry-electrode headband. It is not a
medical device, it has not been reviewed by any regulator, and it must not be
used for diagnosis, treatment or any decision that matters. One frontal channel
cannot read words, thoughts or intentions, and the code says so at every layer
that could otherwise imply it.
