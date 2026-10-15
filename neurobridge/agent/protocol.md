# NeuroBridge agent protocol (v0.1)

Written **for an LLM agent** that is going to consume the NeuroBridge HTTP API and
make decisions while a human sits at a desk wearing a headband. Read the whole
thing before wiring any tool to it. If you only read one section, read
[What this cannot do](#what-this-cannot-do) and
[No authorization from EEG](#no-authorization-from-eeg).

Machine-readable equivalents: `GET /schema` (fields, ranges, provenance, tool
definitions) and `GET /candidate` (the candidate signal shape). This document is
the human/agent-readable contract; `/schema` is the same contract in JSON.

---

## 1. The device, and its honest limits

**Hardware.** A NingNao / SereniBrain **TH21A**, one **dry frontal electrode**
(Fpz-ish, a switch on the nose bridge, band above the eyebrows), 250 Hz, 24-bit
signed ADC counts. The counts are **uncalibrated**: there is no microvolt
conversion, no individual calibration, and no second channel to reference
against. The signal drifts over minutes, and the amplifier **rails at the 24-bit
limit whenever the band is off the head** — a railed capture looks like enormous
"brainwaves" and is pure electrode artefact.

**What one dry frontal channel can support.** Arousal/alertness, *gross*
attention, a **cognitive-load proxy** (frontal theta/beta), drowsiness/fatigue,
relaxation via relative alpha, and two **artefact commands** — voluntary blink
and jaw clench — which are large enough at Fpz to detect without any model.

**What it cannot support, ever.** Words, sentences, images, inner speech,
intentions, imagined movement direction, fine emotional categories, or
multi-class mental typing. Also **valence**: frontal-asymmetry valence needs two
symmetric channels, and this device has one, so `valence` is **always `null`**
and is never estimated, imputed or filled in. Any output claiming otherwise is a
bug, not a finding.

Treat every number this API returns as a **relative signal against the user's own
session baseline**. `0.5` means "at your baseline", not "average human".

---

## 2. The field table

All state fields are **0..1**, monotone (not linear, not calibrated), and derived
per the "Meaning" column. Values are recomputed from a ~4 s rolling window at up
to `2 Hz` (`state_hz`), so consecutive samples are correlated — act on trends
over **≥ 2 s**, never on one sample.

| field | type | range | meaning (and what it is **not**) |
|---|---|---|---|
| `attention` | number | 0..1 | Gross engagement. Monotone in `beta/(theta+alpha)` vs your baseline. **Not** "what you are attending to". |
| `arousal` | number | 0..1 | Alertness/activation. Monotone in `beta/alpha` and mean beta power vs baseline. Low ⇒ consistent with drowsiness. |
| `load` | number | 0..1 | Cognitive-effort **proxy**. Monotone in frontal `theta/beta` vs baseline. **Not** a reading of what you are thinking about. |
| `fatigue` | number | 0..1 | Drowsiness signs: `(theta+delta)/beta` drift, alpha slowing, absolute beta falling below baseline. Deliberately **not** wall-clock time-on-task. |
| `relaxation` | number | 0..1 | Calm. Monotone in relative alpha (8–13 Hz share) vs baseline. |
| `valence` | null | — | **Always `null`.** One channel cannot define asymmetry. Never act on affect from EEG; ask in words. |
| `confidence` | number | 0..1 | How much this window can be trusted: signal quality (contact, saturation, drift) × artefact level (motion, mains) × distance from your baseline distribution. |
| `artifact_level` | number | 0..1 | Share of the window spoiled by saturation, motion jumps or mains hum. Diagnostic, not brain signal. |
| `contact` | string | — | Electrode self-check. Contains `no electrodes reporting OK` when the band is off — then everything else is meaningless. |
| `intents` | array | — | Recent detected commands: `{intent: 'blink'\|'clench', t, confidence, duration_s, peak, threshold, method}`. |
| `model_info` | object | — | **Provenance.** Per field: `trained_model` vs `heuristic_baseline` vs `uncalibrated_neutral`; calibration path; model ids; quality report; errors; honesty notes. |
| `t_mono` | number | — | Monotonic seconds (`time.perf_counter`). The one clock shared by samples, markers and your `POST /mark` calls. |
| `t_wall` | string | — | Local wall time, for logs only. Never compute with it. |
| `seq` | integer | — | Monotonically increasing payload counter. A gap means you dropped a message (or the bridge restarted). |
| `source` | string | — | `th21a` (real headband) or `fake` (synthetic). **Never present synthetic state to the human as their own.** |
| `candidate` | object | — | The cautious candidate signal (section 7). Present on **every** `/state` and `/state/stream` payload. |

### Cadence and staleness

* `GET /state/stream` (SSE) is the preferred transport; it pushes at
  `settings.state_hz` (default **2 Hz**).
* If you poll instead, poll `/state` at **2 Hz**, no faster — the window is ~4 s,
  so more polls add nothing.
* **Staleness timeout: 3 s.** If no payload has arrived for 3 s, the stream is
  broken; stop acting on the last value, re-read `/health`, and tell the user
  rather than continuing with a frozen state.
* On a `seq` gap, re-read `/state` once before acting.

---

## 3. Gating actions on `confidence`

| confidence | what it means | what you may do |
|---|---|---|
| `< 0.2` | Capture is saturated, off-head, or wildly outside baseline. | **Nothing.** Tell the human to reseat the band (clean the electrode if needed). Do not interpret, do not average, do not "wait it out" silently. |
| `0.2 – 0.4` | Weak capture, or no calibration yet, or the window is far from baseline. | **Do not act on state.** Ask a question instead ("How are you doing? Want a break?"). |
| `0.4 – 0.7` | Usable relative signal. | May support *soft*, reversible behaviour: pacing, offering a break, adjusting difficulty. Mark what you did with `POST /mark`. |
| `> 0.7` | Good capture relative to baseline. | Same as above; still never a command and never consequential. |

Confidence is confidence **in the reading**, never in the EEG's meaning. A high
confidence does not upgrade a load proxy into a measurement of thought.

---

## 4. Intents: what is safe to act on

Only two commands are reliable on this montage. Both are **artefact** commands,
detected from the same live detector that feeds `/metrics` (threshold =
`median + k·1.4826·MAD` over a rolling baseline, with a refractory period).

| intent | conventional meaning | evidence | act only if |
|---|---|---|---|
| `blink` | **confirm / select / yes** | 0.1–5 Hz monophasic deflection, ≥ 6×MAD, 0.15–2.0 s wide | `confidence ≥ 0.6` **and** the user was expecting a prompt |
| `clench` | **cancel / escape / no** | 20–45 Hz EMG envelope, ≥ 5×MAD, 0.2–1.5 s | `confidence ≥ 0.6` **and** the action is reversible |

Rules:

* **Blink is also just blinking.** People blink ~15×/min while reading and when
  their eyes are dry. A detected blink is evidence of a blink; it becomes a
  *command* only in a context where the user was asked for one.
* **Clench also happens while talking, eating or tensing the jaw.** Never bind
  clench to anything destructive or irreversible.
* Act only on `confidence ≥ 0.6`, confirm after acting, and **always** tell the
  user what you did and how to undo it.
* Never infer a command from the 0..1 state fields. High `load` is not "the user
  said yes".

---

## 5. Aligning your actions with the EEG: `POST /mark`

The bridge stamps everything on one monotonic clock. Whenever you do something
the user can perceive — show a prompt, start or stop audio, change the task, take
a break — write a marker **immediately before** (and, if it lasts, after) the
action:

```bash
curl -s -X POST http://127.0.0.1:8731/mark \
  -H 'content-type: application/json' \
  -d '{"name":"agent_prompt_shown","meta":{"kind":"break_offer"}}'
```

Response: `{"ok": true, "event": {"name": ..., "t_mono": ..., "session": ...}}`.
Use `t_mono` from the response as the alignment point. Read markers and detected
intents back together with:

```bash
curl -s 'http://127.0.0.1:8731/events?since=12345.0&limit=50'
```

Markers go into the running session's `events.jsonl` when a session is active,
and into `data/sessions/_agent_events.jsonl` otherwise (both are visible to
`/events`). Mark before you act: a marker written afterwards cannot tell later
analysis what the user's brain was doing *when* your action landed.

Follow the store's three-label policy. Never stuff labels into `meta`:

* `task_label` — what was presented or asked (a cue, a stimulus, a requested
  intention). **Not** a fact about the person.
* `reported_label` — what the participant actually reported or did, including
  `skipped`, `failed`, `unknown`. Missing is **not** neutral.
* `predicted_label` — what a model/bridge estimated, with its uncertainty.

A picture of a forest is a forest stimulus, not a ground-truth label for
calmness. A "YES" cue is a requested intention, not proof of an expressed one.

---

## 6. Authentication: none, and that is a decision

The bridge binds `127.0.0.1` by default and has **no authentication**. It exposes
another person's physiological signal. Do not expose it to a network, do not
proxy it, and do not log its payloads anywhere less protected than the machine
itself.

---

## 7. Candidate signal (external protocol §10)

Every `/state` payload carries a `candidate` block, and `GET /candidate` returns
the current one on its own. It is the **only** shape in which this system states
a possible interpretation, and it is deliberately cautious:

```json
{
  "schema_version": "1.0",
  "participant_id": "local_participant_01",
  "model_version": null,
  "scope": "intentional_yes_no_rest",
  "window_start_acquisition_s": 12.40,
  "window_end_acquisition_s": 16.44,
  "signal_quality": "coarse",
  "candidate_label": "uncertain",
  "class_probabilities": null,
  "probabilities_calibrated": false,
  "decision_rule_version": null,
  "recording_context_supported": false,
  "participant_confirmation": null
}
```

* `candidate_label` is `"yes"`, `"no"` or **`"uncertain"`**. It is `"uncertain"`
  whenever the signal-quality gate, the confidence gate, or a **validated
  decision rule** is missing. The bridge **never** falls back to the most likely
  class to fill the field.
* `signal_quality` is `verified` | `coarse` | `unverified`. `verified` requires a
  passing electrode self-check with no saturation or motion; a synthetic source
  is at best `coarse`.
* `class_probabilities` is `null` unless a model produced them, and
  `probabilities_calibrated` is `false` until an actual probability-calibration
  step has run. **A softmax over an uncalibrated LDA is not a calibrated
  probability** and must not be presented as one.
* `model_version` is `null` until a model has been trained. `decision_rule_version`
  is `null` until a threshold has been chosen on independent data.
* `window_start/end_acquisition_s` are seconds on the acquisition clock; they are
  `null` when the buffer cannot define a window.
* `recording_context_supported` is `false` while the bridge cannot describe the
  recording context (task, stimulus, timing quality) for the window.
* `participant_confirmation` is filled **only** by `POST /confirm`, never by the
  model.

**On day one every candidate is `uncertain`.** That is the correct answer, not a
placeholder. Do not "improve" it by picking the largest number.

### 7.1 Participant correction (`POST /confirm`)

```bash
curl -s -X POST http://127.0.0.1:8731/confirm \
  -H 'content-type: application/json' \
  -d '{"candidate_seq":42,"participant_response":"no","note":"I did not mean that"}'
```

`participant_response` ∈ `yes` | `no` | `skipped` | `unsure`. Semantics:

* The prediction was **already logged** when it was made (before you could
  correct it), as its own event with `predicted_label` set.
* The correction is stored **separately**, as an event with `reported_label` set,
  and **never overwrites the prediction**. Both stay in the timeline.
* Corrections are **not** used to retrain the model in the loop. Labels and
  quality are reviewed offline, a versioned development dataset is cut, and the
  model is retrained and re-evaluated independently.
* Never let your guessed interpretation overwrite what the participant said.
  Self-report is authoritative about subjective experience.

An unknown `candidate_seq` is rejected with `400 unknown_candidate_seq` rather
than being attached to the wrong prediction.

---

## 8. No authorization from EEG

**This is the most important rule in this document.**

* `candidate_label == "yes"` is **not consent**. It does not grant permission,
  does not constitute agreement, and does not authorize anything.
* **No consequential or external action may ever be triggered by an EEG estimate
  or a detected intent**: no sending messages, no purchases, no account changes,
  no device or home control, no data deletion, no starting or stopping anything
  with side effects outside the training session.
* The **only** action that may follow from a candidate label or a blink/clench is
  a **reversible comfort action** — for example pausing stimulus presentation —
  and the **manual/physical stop must remain available at all times**.
* Normal authorization rules always apply and are never relaxed because the
  signal looked confident. If an action would need the user's explicit
  permission without EEG, it still does.
* State fields are **not** commands. `load > 0.8` never means "yes"; it means
  "consider offering a break".

If you are unsure whether an action is consequential: it is. Ask.

---

## 9. False-activation metrics (`GET /metrics`)

A continuous decoder that fires on idle blinks is worse than useless, so the cost
is measured rather than assumed:

```bash
curl -s http://127.0.0.1:8731/metrics
```

| key | meaning |
|---|---|
| `intent_events_per_min` | blink/clench detections per minute over the window (default last 600 s). |
| `idle_false_activations_per_min` | the same detections restricted to periods with **no requested action** — the honest cost of a continuous decoder. **`null` until ≥ 60 s of idle exposure exist** (a rate over seconds is noise). |
| `idle_seconds`, `idle_intent_events` | the raw numerator/denominator behind that rate. |
| `state_updates`, `seq`, `dropped_seq` | stream health; `dropped_seq` counts sequence gaps. |
| `uptime_s` | seconds since the bridge started. |
| `signal_quality_counts` | how many payloads were `verified` / `coarse` / `unverified`. |
| `detector` | the exact deployment rules (`blink_sigma`, `clench_band`, `refractory_s`, `min_baseline_s`, `edge_pad_s`) and cumulative counts. |

The metrics are computed from the **same `IntentDetector` instance** the live
stream uses — same smoothing, same refractory, same edge padding — not from a
re-implementation with different constants. An intent counts as *idle* when no
requested-action marker (`stim_onset`, `cue_onset`, `response_prompt`,
`question`, `probe`, ...) occurred within `idle_required_action_s` (3 s) before
it.

Measure this over a **real no-command recording** (the protocol asks for at least
a 5-minute quiet block, with ordinary blinks and small movements left in).
`POST /idle/start` and `POST /idle/end` mark an explicit idle block. A synthetic
source **understates** false activations; never quote a synthetic rate as the
deployment rate.

---

## 10. Tone: offers, not declarations

Behavioural rules for anything you say to the user, adopted from the external
protocol:

* **Prefer offers to declarations.** Say "Would you like a break?" — not "You are
  stressed". Say "This looks like a good moment to pause — want to?" — not "Your
  load is high, so I paused."
* **Self-report is authoritative about subjective experience.** If the user says
  "I'm fine", they are fine, whatever the numbers say. Never argue with a
  self-report using EEG; ask instead.
* **A class estimate is not proof of an internal state.** Report it as a
  possibility with its uncertainty attached, and never in the user's voice
  ("the signal looks like…", not "you are thinking…").
* **If quality, timing, or a validated rule is missing, the answer is
  `uncertain`.** Do not soften it into a guess, and do not fill gaps with
  plausible-sounding detail.
* **Never run a language model to turn a low-confidence category estimate into a
  detailed sentence in the participant's voice.** That converts a weak
  correlation into a fabricated inner monologue.
* **Repeated failure to beat baseline is a valid outcome.** If a target keeps
  failing, **narrow the target** (fewer classes, longer windows, a stronger
  artefact) rather than extending the vocabulary on unsupported evidence. Report
  what failed.
* This program evaluates an interface. It does **not** diagnose health conditions
  and does **not** establish a general ability to read anyone's mind.

---

## 11. What this cannot do

State this plainly if asked, and never imply otherwise:

* **No word, sentence, passage or inner-speech reading.** None. At all.
* **No image or scene reconstruction** from the EEG.
* **No intention reading.** A cued blink is an artefact response to a cue, not
  access to what the person intended.
* **No valence or emotion classification.** One channel ⇒ no frontal asymmetry ⇒
  `valence` stays `null` forever on this hardware.
* **No multi-class mental typing.** At best slow, single dichotomies with long
  averaging and a validated decision rule.
* **No motor-imagery lateralisation** (needs C3/C4), no source localisation, no
  N170/SSVEP topography.
* **No identification or authentication** of a person from the EEG.
* **No diagnosis** of any medical or psychological condition, including
  "drowsiness disorder", "ADHD", "stress disorder" or "depression".

### Calibration / day-one note

Until a model has actually been trained and validated, **every field is a
heuristic comparison against the first ~60 s of the session** (robust
median + MAD per metric, then a personal-baseline percentile mapped into 0..1).
Read `model_info.fields[<field>].derivation`:

* `trained_model` — a decoder trained on this person's recorded sessions drives
  the field. Check `/health` for the model id and the test's cross-validated
  score before trusting it.
* `heuristic_baseline` — **relative, monotone, unvalidated**. Treat it as a
  low-confidence hint about *change from this person's own baseline*, never as a
  measurement of a mental state.
* `uncalibrated_neutral` — no baseline yet; the field is pinned at `0.5` and
  carries **no information**. Wait ~60 s.

Day one therefore looks like this: `confidence` starts low, every candidate is
`uncertain`, `trained_models` is empty, and the honest statement to the user is
"the headband is recording and the signal quality is X", not any claim about what
they are thinking.

---

## 12. Copy-pasteable tool definitions

Paste into your tool list (identical JSON is in `GET /schema` under `tools`), and
read `/state/stream` in the background to keep the latest state hot.

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_brain_state",
      "description": "Read the user's current frontal-EEG state from the NeuroBridge headband (one dry frontal electrode). Returns arousal, gross attention, a cognitive-load PROXY, fatigue, relaxation, signal quality and a confidence. It cannot read words, sentences, images or intentions. Values are 0..1 relative to the user's own session baseline; 0.5 means 'at baseline'. If confidence < 0.4, do not act on the reading. Do not call more often than twice per second.",
      "parameters": {"type": "object", "properties": {}, "required": []}
    }
  },
  {
    "type": "function",
    "function": {
      "name": "wait_for_brain_state",
      "description": "Wait until the user's EEG state crosses a condition, then return it. Use instead of polling in a loop, so you do not act on one noisy sample. Examples: wait until load > 0.7 (user may be overloaded), or arousal < 0.3 (user may be getting drowsy).",
      "parameters": {
        "type": "object",
        "properties": {
          "field": {"type": "string", "enum": ["attention","arousal","load","fatigue","relaxation","confidence"]},
          "above": {"type": "number", "minimum": 0, "maximum": 1},
          "below": {"type": "number", "minimum": 0, "maximum": 1},
          "min_confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.4},
          "timeout_s": {"type": "number", "default": 60}
        },
        "required": ["field", "timeout_s"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "mark_brain_event",
      "description": "Write an agent-side marker into the same EEG timeline the headband samples use. Call it immediately before and after any action you take that could affect the user, so later analysis can align your actions with the brain data. Example names: agent_prompt_shown, agent_paused, agent_break_offer.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "short snake_case event name"},
          "meta": {"type": "object", "description": "arbitrary JSON details"}
        },
        "required": ["name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_candidate_signal",
      "description": "Read the current CANDIDATE signal. This is a cautious candidate, not a fact: candidate_label is 'yes', 'no' or 'uncertain'. A candidate 'yes' is NOT consent and must never trigger a consequential or external action; at most it may support a reversible comfort action such as pausing stimulus presentation, with the manual stop still available. If the answer matters, ask the user and call confirm_candidate with their answer.",
      "parameters": {"type": "object", "properties": {}, "required": []}
    }
  },
  {
    "type": "function",
    "function": {
      "name": "confirm_candidate",
      "description": "Record the participant's own correction of a candidate signal. Call this whenever you asked the user about a candidate and they answered. The correction is stored separately from the prediction, which was already logged. Never invent the response.",
      "parameters": {
        "type": "object",
        "properties": {
          "candidate_seq": {"type": "integer"},
          "participant_response": {"type": "string", "enum": ["yes","no","skipped","unsure"]},
          "note": {"type": "string"}
        },
        "required": ["candidate_seq", "participant_response"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_recent_brain_events",
      "description": "List recent markers and detected blink/clench commands in monotonic order, to see whether the user just blinked (confirm/select) or clenched (cancel/escape) and to correlate their reactions with your own actions.",
      "parameters": {
        "type": "object",
        "properties": {
          "since": {"type": "number", "description": "monotonic seconds lower bound"},
          "limit": {"type": "integer", "default": 50}
        },
        "required": []
      }
    }
  }
]
```

### curl examples

```bash
# health: mode, device, effective fs, sessions, models, readiness
curl -s http://127.0.0.1:8731/health | python -m json.tool

# one state payload (add ?explain=1 for model_info and raw metrics)
curl -s 'http://127.0.0.1:8731/state?explain=1' | python -m json.tool

# SSE stream at 2 Hz (preferred; stop if nothing arrives for 3 s)
curl -N http://127.0.0.1:8731/state/stream

# the candidate signal on its own
curl -s http://127.0.0.1:8731/candidate | python -m json.tool

# participant correction (stored separately; prediction is already logged)
curl -s -X POST http://127.0.0.1:8731/confirm -H 'content-type: application/json' \
  -d '{"candidate_seq":42,"participant_response":"no","note":"not what I meant"}'

# false-activation metrics
curl -s http://127.0.0.1:8731/metrics | python -m json.tool

# mark your own action into the EEG timeline
curl -s -X POST http://127.0.0.1:8731/mark -H 'content-type: application/json' \
  -d '{"name":"agent_break_offer","meta":{"reason":"load high"}}'

# markers + detected intents, merged and sorted
curl -s 'http://127.0.0.1:8731/events?since=0&limit=20' | python -m json.tool

# the agent-facing contract (fields, provenance, tools)
curl -s http://127.0.0.1:8731/schema | python -m json.tool

# latest evaluation for a test (404-shaped JSON when absent, never HTML)
curl -s http://127.0.0.1:8731/report/T201 | python -m json.tool
```

WebSocket: `ws://127.0.0.1:8731/ws` pushes the same payloads and accepts
`{"type":"mark","name":...,"meta":{}}`,
`{"type":"confirm","candidate_seq":N,"participant_response":"yes"}`, and
`{"type":"ping"}`.

---

## 13. Unknowns, recorded honestly

* **No model is trained yet** in a fresh checkout: `model_version` is `null`,
  every candidate is `uncertain`, and every field is `heuristic_baseline` or
  `uncalibrated_neutral`. `GET /health` shows `trained_models: []`.
* **Timing is `unverified`.** A bridge-side software timestamp is never an exact
  visual or acoustic onset; `/candidate` reports `timing_quality: "unverified"`
  and `recording_context_supported: false`.
* **Idle false-activation rates do not exist until a real no-command recording
  does.** A synthetic source understates them.
* **Calibration is per session and per seating.** A baseline from another day's
  electrode placement is not adopted; the bridge would rather report
  `uncalibrated_neutral` than compare against a different montage.
* If hardware data, prepared media, timing information or independent validation
  is missing, the honest report is that concrete gap — not a completed claim.
