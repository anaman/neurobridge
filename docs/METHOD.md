# Method: what is measured, how, and why it is measured that way

This document is the reasoning behind the catalog, the pipeline and the gates.
If a result ever looks surprising, the answer is usually here.

## 1. The measurement problem on one dry frontal electrode

Four properties of this hardware dictate every analysis choice:

1. **No spatial information.** One electrode cannot separate a frontal P300 from
   frontal eye movement, or mu rhythm from jaw EMG, by topography. Separation
   has to come from *time*, *frequency* and *experimental design* instead.
2. **No calibration.** Raw counts have no microvolt meaning, and the gain is
   unknown and possibly temperature-dependent. Therefore: relative band power,
   log-ratios, and within-session z-scores against the subject's own baseline.
   Absolute amplitude is never reported as a physiological quantity.
3. **Non-stationarity.** Dry electrodes polarise and drift over minutes; contact
   changes when you move. Therefore: short blocks, per-session baselines,
   per-block normalisation, and grouped cross-validation.
4. **Artefacts that are larger than the signal.** A blink is roughly 5–20× the
   amplitude of a frontal P300. In the state and ERP tests these are removed; in
   the tier-7 control tests the very same artefacts *are* the signal, and the
   specs raise their rejection thresholds accordingly.

## 2. Why the tiers are ordered the way they are

```
Tier 0  can we trust the channel at all?          (contact, latency, artefacts, null floor)
Tier 1  coarse whole-brain state                  (arousal, alertness, fatigue)
Tier 2  cognitive state                           (load, attention, effort)
Tier 3  time-locked auditory responses            (P300, MMN)  <- the cleanest evoked signals
Tier 4  time-locked visual/semantic responses     (oddball, N400) <- weaker frontally
Tier 5  sustained language processing             (reading)
Tier 6  sustained auditory/music processing
Tier 7  volitional control + closed loop with the agent
```

Tiers 0–1 are prerequisites, not warm-ups. A session in tier 3 with bad contact
produces nothing, and the analysis cannot tell "no P300" from "no electrode".

## 3. Band-power features (tiers 1, 2, 5, 6, 7)

Welch PSD (Hann, 50 % overlap, n_fft = 2 s) over 1–45 Hz, then:

* `relbandpower:<band>` — % of 1–45 Hz power. Scale-free, so it survives gain
  and contact changes far better than absolute power.
* `ratio:<a>_<b>` — log10 of two band powers. The canonical examples are
  theta/beta (a cognitive-effort proxy) and beta/alpha (an arousal proxy).
* `aperiodic:exponent` — the 1/f slope, fitted over 2–40 Hz with the alpha peak
  excluded. This matters because a change in "theta power" can be a change in
  the aperiodic component rather than a change in an oscillation, and on a
  single channel those are easy to confuse.
* `peakfreq:dominant`, `entropy:spectral`, Hjorth parameters — cheap
  descriptors of the spectral shape that catch drowsiness (slowing) and
  artefact (flattening) respectively.

Normalisation: every windowed feature is z-scored against the first ~60 s of the
session (median/MAD, not mean/sd — robust to the occasional blink burst).
Window length is 4 s with a 1 s step: long enough that a 10 Hz alpha rhythm
contributes ~40 cycles to the estimate, short enough that a 3-minute block still
yields ~150 windows.

## 4. ERP features (tiers 3, 4)

Epochs are `stim_onset`-locked, baseline-corrected on the pre-stimulus window,
and rejected when peak-to-peak exceeds the spec's threshold or when any sample
is railed. Then, per epoch: mean and peak amplitude in named windows —
`N1` 80–180 ms, `P2` 150–250 ms, `P3`/`P300` 250–450 ms, `N400` 300–500 ms,
`MMN` 100–250 ms.

The single most important design rule for these tiers: **jittered SOA**. With a
fixed 1.0 s SOA the ongoing alpha and the evoked response are phase-locked to
the same clock, and the "P300" can be an alpha artefact. Every ERP spec in the
catalog therefore specifies `iti_jitter_s`.

## 5. How many trials? (a real power calculation, not a guess)

For a frontal dry electrode, a realistic single-trial P300 amplitude in this
setup is small relative to ongoing EEG. Take a conservative single-trial effect
size of d ≈ 0.3 (3 µV response against ~10 µV single-trial noise). For 80 %
power at α = 0.05 (two-sided):

```
n per group ≈ 2 (z_{1-α/2} + z_{1-β})² / d²  ≈  2 (1.96 + 0.84)² / 0.09  ≈  174
```

So a 160-standard / 40-deviant oddball (T301, 200 trials) sits right at that
boundary — which is exactly why T301 is the most reliable ERP test in the
catalog and why the tiers above it use *sustained* designs where a 4 s window
integrates over many cycles rather than a 0.7 s epoch.

This is also why several tier-4 and tier-5 tests are marked `exploratory`: their
expected single-trial effect size is nearer d ≈ 0.1, which would need ~1500
trials per class for the same power. They are worth recording as a dataset, but
their gates are set near chance and their reports say so.

**Consequence for the decoder.** With 40–160 trials per class and features in
the tens, a flexible model will overfit. The catalog therefore defaults to LDA
or logistic regression on a small, pre-registered feature set. EEGNet and the
Riemannian classifier exist and are used where the spec asks for them, but the
default is deliberately the least flexible thing that can work.

## 6. Cross-validation: why blocks, not trials

Trials inside a block share a slow drift, a contact state, a posture and a
fatigue level. Random k-fold lets the classifier see the same drift in train and
test, which inflates accuracy by a lot on non-stationary data. Every reported
number in this system therefore comes from **leave-one-block-out** (or grouped
k-fold when there are many blocks). Where cross-session data exists,
`nb train --all-sessions` evaluates the harder and more honest question: does
the decoder survive a new day with a new electrode contact?

## 7. Gates, chance and the null floor

Each spec declares its gates *before* recording. Gates are set per feasibility
band:

| band | balanced accuracy | auc | what a pass means |
|---|---|---|---|
| established | 0.70–0.80 | 0.75–0.85 | a real, replicated effect we can rely on |
| promising | 0.60–0.65 | 0.65–0.70 | a real but modest effect |
| exploratory | 0.55–0.58 | 0.60 | barely above chance by design; a dataset contribution |
| control | ≤ 0.60 | ≤ 0.55 | must *fail* — this is the pipeline's false-positive floor |

Three protections against fooling ourselves:

1. **Chance is always printed next to accuracy**, so a 0.62 on 3 classes
   (chance 0.33) can never be confused with a 0.62 on 2 classes (chance 0.50).
2. **Permutation testing** (`permutation_test`, 200 label permutations *within
   blocks*) turns "0.61 > 0.58" into a p-value.
3. **`nb simulate --effect-size 0`** regenerates a session with the injected
   response switched off. If that decodes above chance, the leakage is in this
   code, not in the brain.

## 8. Timing: why the marker offset is measured, not assumed

Stimulus onset and EEG sample times come from the same `perf_counter()` clock,
but three delays sit between "the CPU decided to show the image" and "the
response appears in the recorded samples": display flip latency, audio buffer
latency, and BLE transport latency. Summed, they are typically tens of
milliseconds and they are *systematic*, which is exactly the kind of error that
silently destroys an ERP latency measurement while leaving amplitude intact.

`T004` (click train) and `T005` (flash train) exist to measure the total offset
by cross-correlating the averaged evoked response: the click is known to evoke
an early auditory response ~40–120 ms after the true onset, so the measured peak
latency minus that physiological latency is the pipeline offset. The value is
stored per machine in `output_latency_offset_s`.

## 9. What "training the translator" means, concretely

Two stages, and the order is not negotiable.

**Stage 1 — state telemetry (tiers 1, 2, 5, 6, 7).** The agent receives
`attention`, `arousal`, `load`, `fatigue`, `relaxation` and an honest
`confidence`, each z-scored against your baseline. There is nothing to "type":
the agent adapts (paces itself, shortens answers when load is high, offers a
break when fatigue rises). This stage can be useful on day three, because it
needs only band power and a baseline.

**Stage 2 — discrete commands (tier 7).** Blink and jaw clench are large,
reliable, and detectable at single-trial level; they give a confirm and a cancel
with high accuracy on the first day. Beyond that, every additional command costs
dramatically more averaging: a two-class ERP switch needs ~40–160 trials *per
decision*, i.e. tens of seconds per bit. That is the true ceiling of this
hardware, and it is why the catalog labels the rest `exploratory` rather than
promising a mental keyboard.

## 10. Behavioural channels and the labels they produce

Two non-EEG channels are recorded on the same monotonic clock, and both are
governed by the same rule:

| Channel | Measures | Never |
|---|---|---|
| Voice (`--mic`) | cue-to-speech onset latency, speech duration, rate proxy, pauses, level | a cortical latency, a word rate, or an EEG feature |
| Self-report (post-trial) | what the participant says happened | a neutral default when they skipped |

The rule, adopted from the imported external protocol
(`docs/PROTOCOL_MAPPING.md`), is that a trial has three separable labels and
they must not be conflated:

* `task_label` — what was presented or requested. A picture of a forest is a
  forest stimulus, not a claim that you felt calm.
* `reported_label` — what you actually reported or did, including *skipped*,
  *unknown* and *failed*. Missing is not neutral and is never imputed.
* `predicted_label` — what a model estimated from eligible EEG, always with
  signal-quality and uncertainty attached, and `uncertain` when a validated
  decision rule, sufficient quality or sufficient confidence is absent.

**The leakage rule for behavioural channels.** Voice and self-report metrics must
never enter the EEG feature matrix. A classifier that predicts "spoke aloud"
from features correlated with the microphone has learned the microphone. They
may be used in exactly two ways: as the *target* of a separate question ("does
frontal theta predict speech onset latency?") or as a *confound check* ("do the
audio envelope and gamma power track each other across trials?"). `T502` is the
test where this matters most, because articulation is simultaneously the signal
of interest and the largest artefact at Fpz.

## 11. Reproducibility

* Every session stores `meta.json` with the device info, settings snapshot, fs,
  montage, spec id, and — for synthetic runs — the seed, effect size and
  duration scale.
* Every analysis is a pure function of (session data, spec YAML). Re-running
  `nb eval` on the same session gives the same number.
* The synthetic path uses fixed seeds, so `tests/test_smoke.py` is a real
  regression test for the whole chain: acquisition format, markers, epoching,
  features, decoder and gates.
