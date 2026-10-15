# Crosswalk: the Codex protocol (T01–T64) ↔ NeuroBridge (T001–T708)

Two independently written 64-test designs now live in this repository:

| | **Codex protocol** | **NeuroBridge catalog** |
|---|---|---|
| File | [`EEG_AGENT_TRAINING_PROTOCOL.md`](EEG_AGENT_TRAINING_PROTOCOL.md) (verbatim) | [`../neurobridge/catalog/tests/`](../neurobridge/catalog/tests) |
| Machine-readable | [`../neurobridge/catalog/protocol_codex.yaml`](../neurobridge/catalog/protocol_codex.yaml) | one YAML per test |
| Ids | `T01`–`T64`, modules A–H | `T001`–`T708`, tiers 0–7 |
| Unit | one **test card** = one stimulus type + instruction + self-report | one **spec** = paradigm + markers + features + decoder + gates |
| Organising question | *what should be shown and what should be asked?* | *what signal can be extracted, and would we know if it were not there?* |
| Status | specification. No code, no recordings, no claim that a label is decodable | runnable implementation with synthetic end-to-end verification |

They are complementary, not competing: the Codex protocol is a **content and
epistemics battery**, the NeuroBridge catalog is the **acquisition and analysis
machinery**. This document says exactly how they line up, and where they do not.

## Status vocabulary

| Status | Meaning |
|---|---|
| `covered` | NeuroBridge already runs a protocol that measures the same thing |
| `host` | the Codex *content* has no NeuroBridge equivalent, but an existing spec is the right paradigm to present it in |
| `gap` | neither the content nor an equivalent paradigm exists in NeuroBridge yet |
| `eval` | it is an evaluation design, not a stimulation test; it maps onto tooling rather than a spec |

## The one thing that must not be lost

The Codex document states it plainly and it governs everything below:

> The numbered items are **64 distinct test types**, not 64 samples sufficient to
> train a translator. Each usable repetition is a trial.

Every card in modules B–G is written as **one 8–13 s exposure**. One exposure per
concept cannot train a classifier, and a classifier evaluated on the same
exposure it was fitted on proves nothing. NeuroBridge's specs therefore
pre-register trial counts and jittered inter-trial intervals, and its report
layer reports accuracy against chance on **held-out blocks**.

The practical consequence: the Codex cards define *what each trial looks like*;
they must be **repeated 40+ times per class** (see `docs/METHOD.md` §5 for the
power calculation) and hosted inside a NeuroBridge paradigm before any decoding
claim is attempted. Nothing in either document changes that arithmetic.

---

## Module A — recording quality and artifacts → Tier 0

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T01 | Eyes-open fixation baseline | `T002` resting_eyes_open_baseline | covered |
| T02 | Eyes-closed baseline | `T003` alpha reactivity (open vs closed) | covered |
| T03 | Blink artifact examples | `T007` blink/clench artifact library | covered |
| T04 | Gaze-shift artifact examples | — | **gap** (see G1) |
| T05 | Jaw-muscle artifact examples | `T007` (clench condition) | covered |
| T06 | Actual left-finger movement | `T107` artifact census (as artifact) | host |
| T07 | Actual right-finger movement | `T107` artifact census (as artifact) | host |
| T08 | Unstructured quiet baseline | `T002` / `T107` | covered |

**Honest note on T06/T07.** A single dry *frontal* electrode cannot lateralise
finger movement; there is no C3/C4 and no contralateral topography. These cards
are still useful — as *movement-artifact* calibration, which is how they are
hosted. Any above-chance left/right result here is more likely to be muscle,
eye movement or effort than a motor signal, and `T704` (motor imagery) says the
same thing about itself.

## Module B — reading and text controls → Tiers 0/5

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T09 | Read about water | `T501` self-paced reading (host) | host |
| T10 | Read about fire | `T501` (host) | host |
| T11 | Read about a dog | `T501` (host) | host |
| T12 | Read about a piano | `T501` (host) | host |
| T13 | Literal text comprehension | `T205` comprehension vs mindless reading | covered |
| T14 | Simple text inference | `T205`, `T504` garden-path | covered |
| T15 | Text with negation | `T404` N400 congruency (host) | **gap** (see G2) |
| T16 | Reading command words control | `T406` lexical decision (as control) | host |

## Module C — images → Tiers 1/4

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T17 | Observe image: water | `T401` visual oddball / `T402` RSVP | host |
| T18 | Observe image: fire | `T401` / `T402` | host |
| T19 | Observe image: dog | `T401` / `T403` face-vs-object | host |
| T20 | Observe image: piano | `T401` / `T402` | host |
| T21 | Observe and rate: forest path | `T408` affective image valence | host |
| T22 | Observe and rate: city street | `T408` | host |
| T23 | Observe and rate: flowers | `T408` | host |
| T24 | Observe and rate: hand tools | `T408` | host |

**Where Codex is stricter than NeuroBridge.** `T408` sorts images into
`pleasant` / `unpleasant` / `neutral` *a priori*. The Codex label policy is
better: the target should be **the participant's own rating**, recorded
afterwards, not a universal emotional label attached to a picture. A forest is
a forest stimulus. `T408`'s notes now carry that caveat; adopting the rating as
the label requires the `reported_label` field (see N1 below).

## Module D — Freesound listening → Tier 6

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T25 | Listen: water | `T601` music/speech/noise, `T606` | host |
| T26 | Listen: fire | `T601`, `T302` environmental oddball | host |
| T27 | Listen: dog | `T601`, `T302` | host |
| T28 | Listen: piano | `T601` (music condition) | host |
| T29 | Listen and rate: birds | `T606` nature vs urban | host |
| T30 | Listen and rate: traffic | `T606` | host |
| T31 | Listen and rate: rain | `T606` | host |
| T32 | Listen and rate: machine hum | `T606`, `T607` loudness ladder | host |

## Module E — meaning across modalities → Tier 4

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T33 | Text-image agreement: water/water | `T405` picture-word match | covered |
| T34 | Text-image agreement: water/fire | `T405` (mismatch condition) | covered |
| T35 | Text-image agreement: fire/fire | `T405` (match) | covered |
| T36 | Text-image agreement: fire/water | `T405` (mismatch) | covered |
| T37 | Image-sound agreement: dog/dog | `T405` (audio variant) | host |
| T38 | Image-sound agreement: dog/piano | `T405` (audio variant) | host |
| T39 | Image-sound agreement: piano/piano | `T405` (audio variant) | host |
| T40 | Image-sound agreement: piano/dog | `T405` (audio variant) | host |

This is the cleanest overlap between the two designs: `T405` is exactly the
match/mismatch paradigm the Codex module describes, and its features
(`erp:N400:300:500`, `erp:P3:250:450`) are the right ones for it. T37–T40 extend
it from text-image to image-sound agreement, which `T405` does not yet support.

## Module F — task demand and memory → Tier 2

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T41 | Single-step mental arithmetic | `T202` arithmetic vs counting | covered |
| T42 | Two-step mental arithmetic | `T202` (hard condition) | covered |
| T43 | Retain three digits | `T201` n-back (host) | **gap** (see G3) |
| T44 | Retain six digits | `T201` (host) | **gap** (G3) |
| T45 | One-back comparison | `T201` n-back 0 vs 2 (as one-back) | covered |
| T46 | Two-back comparison | `T201` (two-back condition) | covered |
| T47 | Apply one decision rule | `T201` (host) | **gap** (G4) |
| T48 | Apply combined decision rules | `T201` (host) | **gap** (G4) |

Digit span (T43/T44) and rule complexity (T47/T48) are *different*
manipulations from n-back: span taxes maintenance without updating, and rule
complexity taxes selection without memory load. Hosting them on `T201` works
mechanically but the resulting model would be "demand detector", not "n-back
detector" — the spec's `goal` and the report must say which was trained.

## Module G — intentional commands → Tier 7

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T49 | Intend YES | `T508` inner speech (closest recorded paradigm) | **gap** (see G5) |
| T50 | Intend NO | `T508` (closest) | **gap** (G5) |
| T51 | Intend acceptance, alternate wording | — (generalisation design) | **gap** (G5) |
| T52 | Intend refusal, alternate wording | — | **gap** (G5) |
| T53 | Intend MORE | — | **gap** (G5) |
| T54 | Intend PAUSE | — | **gap** (G5) |
| T55 | No-command control | `T008` sham (statistical null) | **eval** (see N2) |
| T56 | Free choice without a class cue | — | **gap** (G5) |

**This is the module where the two designs disagree most, and the disagreement
is worth stating plainly.** NeuroBridge's tier 7 deliberately implements the
only commands this hardware supports reliably: **blink** and **jaw clench**,
which are large single-trial artefacts at Fpz, plus a cued confirm/cancel
handshake (`T706`). The Codex cards T49–T56 instead ask for *covert intentions
with no motor component*.

On one frontal dry channel, a cued covert YES/NO intention is an `exploratory`
target at best: the literature effect is small, the artefact confounds (eye
movement, subtle jaw and brow activity, breathing) correlate with exactly the
internal state, and the required averaging is tens of seconds per decision. The
NeuroBridge position is that this must be *attempted and reported honestly*,
not promised — which is why it is a gap rather than a spec: adding it means
writing a spec whose gate sits at 0.55–0.58 and whose risks name the confounds
first. That is a legitimate next step, not something to fake now.

## Module H — reserved evaluation → evaluation tooling

| Codex | Card | NeuroBridge | Status |
|---|---|---|---|
| T57–T58 | Held-out reading: water, fire | `nb train` (dev sessions) → `nb eval` (held-out session) | eval |
| T59–T60 | Held-out image: dog, piano | same | eval |
| T61–T62 | Held-out sound: water, fire | same | eval |
| T63–T64 | Held-out intention cue: YES, NO | same | eval |

Module H is not a stimulation protocol; it is a **split discipline**. It is the
module with the largest real gap in NeuroBridge (N3 below): the pipeline does
grouped cross-validation by block, but there is no *frozen split manifest* that
separates development from locked evaluation before training, and no way to
declare "these sessions are locked; never fit on them".

---

## Gaps and next steps

| # | Gap | Where it belongs | Effort |
|---|---|---|---|
| G1 | Gaze-shift artifact card (T04) | extend `T007` with a `gaze_left` / `gaze_right` / `gaze_center` condition, or add `T009` | small |
| G2 | Negation text control (T15) | new spec hosting negation vs affirmative frames on the `T404` N400 recipe | small |
| G3 | Digit span 3 vs 6 (T43/T44) | new sustained spec, `load_low`/`load_high` classes, same features as `T201` | small |
| G4 | Rule complexity 1 vs 2 rules (T47/T48) | new sustained spec, same recipe as `T201` | small |
| G5 | Cued intentional commands YES/NO/MORE/PAUSE + free choice + alternate wording (T49–T54, T56) | new tier-7 specs at `exploratory` gates, plus a generalisation split (T51/T52) | large — needs 40+ trials per class per session |
| N1 | `task_label` / `reported_label` / `predicted_label` separation, and "missing is not neutral" | `store.py` (event schema) + `agent/` (output contract) | implemented in this import |
| N2 | False activations per minute during no-command idle | `pipeline/evaluate.py` + the agent bridge report | implemented in this import (metric), needs a live idle recording |
| N3 | Frozen split manifest separating development from locked evaluation | new `pipeline/splits.py` + `nb split` / `nb eval --locked` | implemented in this import |

## What was adopted verbatim, and what was not

**Adopted and enforced in code by this import**

1. The **three-field label policy** — a stimulus label is not a ground truth
   about the person, a requested intention is not evidence of an expressed one,
   and a correct answer is not a neural mechanism. Events now carry
   `task_label`, `reported_label` and `predicted_label` separately, and the
   agent output contract exposes a *candidate* label with `uncertain`.
2. **"Missing is not neutral"** — a skipped or failed trial is recorded as such
   and never silently imputed.
3. **No external actions from an EEG estimate.** An EEG candidate label never
   grants permission; the agent bridge documents that a predicted command is
   not consent, and only reversible comfort actions (pause) may follow from it.
4. **Timing honesty** — `timing_quality` is recorded as `verified`, `coarse` or
   `unverified`; a bridge-side software timestamp is never called an exact
   visual or acoustic onset.
5. **The media ledger** — the 20 candidate assets are imported with their
   licences, attributions, Openverse ids, source pages and media URLs, so they
   can be fetched, hashed and credited rather than re-searched.

**Not adopted**

* The Codex id scheme is not merged into the NeuroBridge catalog: `T01` and
  `T001` differing by one character is a bug waiting to happen. The two sets
  stay in separate files with separate loaders.
* Codex's proposed `50/60 Hz notch` caution is already handled — NeuroBridge
  measures mains contamination and reports it rather than always notching.
* No claim in the Codex document is upgraded. Where it says "success is
  uncertain", NeuroBridge keeps the corresponding gate at chance-plus-epsilon.

## Reproducing this import

```bash
.venv/bin/python scripts/import_protocol.py          # verbatim copy + parse
.venv/bin/python scripts/import_protocol.py --force  # refresh the copy
```

The copy is byte-identical to the source (`docs/EEG_AGENT_TRAINING_PROTOCOL.provenance.json`
records the sha256), and the parser refuses to drop a field it does not
recognise — unknown bullets are preserved under `extra:` rather than lost.
