# Catalog specification — the 64 tests

Every test is one YAML file in `neurobridge/catalog/tests/` named
`<ID>_<slug>.yaml`. `nb catalog validate` checks all of them against
`neurobridge/catalog/schema.py`; a spec that fails validation never runs.

## Hardware reality (read this first)

The acquisition channel is **one dry frontal electrode** (switch on the nose
bridge, band above the eyebrows, 250 Hz, 24-bit counts). Consequences that
every spec must respect:

| Possible | Not possible |
|---|---|
| Alpha/theta/beta band-power states, alpha reactivity | Source localisation |
| Frontal P300 / MMN (reduced amplitude vs Pz) | Posterior N170 / SSVEP with clean topography |
| Blink and jaw-clench artifact control (these are *strong* at Fpz) | Motor imagery lateralisation (needs C3/C4) |
| Cognitive-load markers (theta/beta, frontal midline theta) | Fine-grained semantic decoding of words |
| Arousal/drowsiness tracking | Reading thoughts as sentences |
| Single dichotomies and slow 2–4 way choices with long averaging | Fast multi-class mental keyboard |

`feasibility` must therefore be honest: `established`, `promising`,
`exploratory`, or `control`. An `exploratory` spec is legitimate — it just has
a lenient gate and says so in the report.

## Required fields

```yaml
id: T301                       # T<4 digits>, unique, range fixed per tier
slug: auditory_oddball_p300    # snake_case
tier: 3                        # 0..7, must match the id's hundred digit
title: "Auditory oddball (P300)"
modality: [audio]              # text|image|audio|multimodal|none
feasibility: established
goal: >
  One or two sentences: what state/signal this test is meant to expose and
  what the agent will do with it.
classes: [standard, deviant]   # 1..6 labels; keep to 2-3 unless justified
stimuli:
  source: synth                # openverse|freesound|synth|textbank|none|mixed
  kind: audio                  # image|audio|text|tone
  queries:                     # REQUIRED when source is openverse|freesound|mixed
    standard: ["1000 hz tone"]
    deviant: ["sine sweep"]
  n_per_class: 2               # items per class fetched/generated
  duration_s: 1.2              # audio/tone/RSVP presentation length
  license: ["cc0", "by"]       # Openverse license filter
  fallback: synth              # synth|textbank|builtin|none  (used offline or without a token)
design:
  trials: 200
  duration_min: 8
  blocks: 4
  soa_s: 1.2                   # stimulus onset asynchrony
  iti_jitter_s: [0.3, 0.6]     # REQUIRED for ERPs: jitter prevents entrainment
  p_target: 0.2                # only where a class is rare
  rest_between_blocks_s: 30
markers:
  block_start: 1300
  block_end: 1301
  stim_onset: 1302             # must appear in processing.onset_markers
  standard: 1310               # marker key == class label (lowercased, _ for spaces)
  deviant: 1311
processing:
  fs: 250
  onset_markers: [stim_onset]
  bandpass: [0.3, 30]
  epoch: [-0.2, 0.8]           # must straddle 0
  baseline: [-0.2, 0.0]
  artifact_ptp: 400000         # reject epochs above this peak-to-peak (counts)
  reject_policy: reject        # reject|flag|none
features: [erp:P3:250:450, erp:N1:80:180, bandpower:theta, ratio:theta_beta]
decoder: {model: lda, cv: leave_one_block_out, normalize: zscore}
metrics: [balanced_accuracy, auc, permutation_p]
acceptance: {balanced_accuracy: 0.65, auc: 0.70}
risks:
  - "single frontal channel -> P300 is 2-4x smaller than at Pz"
  - "dry electrode drift over an 8 minute block"
agent_use: none                # none|state|command|both
```

### Marker code allocation

Per-test codes must be unique **within the file** and inside the tier band:

| tier | band | tier | band |
|---|---|---|---|
| 0 | 1000–1099 | 4 | 1400–1499 |
| 1 | 1100–1199 | 5 | 1500–1599 |
| 2 | 1200–1299 | 6 | 1600–1699 |
| 3 | 1300–1399 | 7 | 1700–1799 |

Within a band: `+0..+9` phase markers (`block_start`, `block_end`, `cue_onset`,
`response_prompt`, `feedback`, `break_start`, `break_end`), `+10..+49` class
markers (one per class, key == class label), `+50` `stim_onset`, `+51..+59`
subsidiary events (`blink`, `clench`, probe, question). Always define
`stim_onset`; it is the epoch trigger.

### Feature vocabulary (implemented by `pipeline/features.py`)

| pattern | meaning |
|---|---|
| `bandpower:<band>` | log10 absolute power, band ∈ delta theta alpha low_beta beta high_beta gamma |
| `relbandpower:<band>` | % of 1–45 Hz power |
| `ratio:<a>_<b>` | log10(a / b) of two band powers |
| `erp:<name>:<lo_ms>:<hi_ms>` | mean amplitude in a post-stimulus window |
| `erp_peak:<name>:<lo_ms>:<hi_ms>` | peak amplitude in a window |
| `hjorth:<mobility\|complexity>` | Hjorth parameters |
| `entropy:spectral` | spectral entropy |
| `stats:<std\|ptp\|mean\|skew\|kurt\|zcr>` | time-domain statistics |
| `slope:mean` | mean absolute first difference |
| `peakfreq:dominant` | dominant frequency 1–45 Hz |
| `aperiodic:exponent` | 1/f slope (FOOOF-lite) |
| `cov:tangent` | covariance for Riemannian decoders |

### Decoders

`decoder.model` ∈ `lda | logreg | svm | riemann | eegnet`.
`decoder.cv` ∈ `leave_one_block_out | stratified_group_kfold | stratified_kfold`.
Sustained-state tests (band-power features) should prefer `lda`/`riemann`;
ERP tests `lda` with `erp:*` features; anything with `cov:tangent` needs
`riemann`.

### Acceptance gates

Numeric gates on `metrics`. Reference values to use:

| feasibility | balanced_accuracy | auc | note |
|---|---|---|---|
| established | 0.70–0.72 | 0.75 | e.g. blink switch, alpha reactivity |
| promising | 0.60–0.65 | 0.65–0.70 | |
| exploratory | 0.55–0.58 | 0.60 | barely above chance, honest |
| control | 0.50–0.60 | 0.55 | must NOT exceed 0.75 (schema enforces) |

Chance level for k balanced classes is `1/k`; the report always prints the
gap over chance, so gates must be above chance but attainable.

## The 64 tests

### Tier 0 — Commissioning & signal integrity (established: 6, control: 1)

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T001 | `electrode_seating_selfcheck` | Electrode seating & self-check | established | good_contact, poor_contact | none |
| T002 | `resting_eyes_open_baseline` | Resting baseline, eyes open | established | eyes_open | none |
| T003 | `resting_eyes_closed_alpha_reactivity` | Alpha reactivity (eyes open vs closed) | established | eyes_open, eyes_closed | none |
| T004 | `auditory_latency_calibration` | Auditory latency calibration (click train) | established | click | none |
| T005 | `visual_latency_calibration` | Visual latency calibration (flash train) | promising | flash | none |
| T006 | `mains_and_drift_characterisation` | Mains, drift & saturation characterisation | established | baseline | none |
| T007 | `blink_clench_artifact_library` | Blink & jaw-clench artifact library | established | blink, clench, rest | none |
| T008 | `sham_marker_control` | Sham marker control (null floor) | control | sham | none |

### Tier 1 — Arousal, alertness & fatigue

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T101 | `sustained_attention_sart` | Sustained attention (SART) | promising | target, lapse | state |
| T102 | `drowsiness_induction_detection` | Drowsiness induction & detection | promising | alert, drowsy | state |
| T103 | `alpha_entrainment_flicker` | Alpha entrainment to flicker | exploratory | flicker_8, flicker_10, flicker_12 | none |
| T104 | `arousal_audio_calm_vs_exciting` | Arousal from audio (calm vs exciting) | promising | calm, exciting | state |
| T105 | `vigilance_decrement_time_on_task` | Vigilance decrement over time-on-task | promising | early, late | state |
| T106 | `caffeine_state_comparison` | Pre/post state comparison | exploratory | pre, post | none |
| T107 | `artifact_census_resting` | Artifact census during rest | established | rest | none |
| T108 | `microsleep_probe_prolonged_closure` | Microsleep probe under eye closure | exploratory | responded, missed | state |

### Tier 2 — Attention & cognitive load

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T201 | `nback_0_vs_2` | N-back 0 vs 2 (working memory load) | promising | load_low, load_high | state |
| T202 | `mental_arithmetic_vs_counting` | Mental arithmetic vs counting | promising | arithmetic, counting | state |
| T203 | `stroop_colour_word` | Stroop colour-word interference | promising | congruent, incongruent | state |
| T204 | `visual_search_easy_vs_hard` | Visual search, easy vs hard | promising | easy, hard | state |
| T205 | `reading_comprehension_vs_mindless` | Comprehension vs mindless reading | promising | comprehension, mindless | state |
| T206 | `speech_in_noise_snr_ladder` | Speech in noise (SNR ladder) | promising | snr_easy, snr_hard | state |
| T207 | `dual_task_load` | Dual-task load | exploratory | single, dual | state |
| T208 | `focused_attention_vs_mind_wandering` | Focused attention vs mind-wandering | exploratory | focused, mind_wandering | state |

### Tier 3 — Auditory event-related potentials

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T301 | `auditory_oddball_p300` | Auditory oddball (P300) | established | standard, deviant | none |
| T302 | `environmental_oddball` | Environmental-sound oddball | promising | standard, deviant | none |
| T303 | `mmn_duration_deviant` | MMN, duration deviant | promising | standard, deviant | none |
| T304 | `mmn_frequency_passive_ignore` | MMN, passive frequency deviant | established | standard, deviant | none |
| T305 | `speech_syllable_oddball` | Speech-syllable oddball (/ba/ vs /da/) | exploratory | standard, deviant | none |
| T306 | `roving_standard_oddball` | Roving-standard oddball | promising | standard, deviant | none |
| T307 | `own_name_vs_other_name` | Own name vs other name | exploratory | own_name, other_name | none |
| T308 | `auditory_stream_segregation` | Auditory stream segregation | exploratory | segregated, integrated | none |

### Tier 4 — Visual & semantic event-related potentials

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T401 | `visual_oddball_openverse` | Visual oddball with Openverse images | promising | standard, deviant | none |
| T402 | `rsvp_natural_vs_manmade` | RSVP natural vs man-made | promising | natural, manmade | none |
| T403 | `face_vs_nonface_n170` | Face vs non-face (N170 surrogate) | exploratory | face, object | none |
| T404 | `n400_sentence_congruency` | N400 sentence congruency | promising | congruent, incongruent | none |
| T405 | `n400_picture_word_match` | N400 picture-word match | exploratory | match, mismatch | none |
| T406 | `lexical_decision` | Lexical decision (word vs pseudoword) | promising | word, pseudoword | none |
| T407 | `alpha_blocking_image_luminance` | Alpha blocking by image luminance | promising | bright, dark | state |
| T408 | `affective_image_valence` | Affective image valence | exploratory | pleasant, neutral, unpleasant | none |

### Tier 5 — Reading & language

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T501 | `self_paced_silent_reading` | Self-paced silent reading | promising | reading | state |
| T502 | `reading_aloud_vs_silent` | Reading aloud vs silent | promising | aloud, silent | state |
| T503 | `word_frequency_effect` | Word-frequency effect | promising | high_freq, low_freq | none |
| T504 | `garden_path_sentences` | Garden-path vs simple sentences | exploratory | simple, garden_path | none |
| T505 | `prose_vs_shuffled_words` | Coherent prose vs shuffled words | promising | prose, shuffled | state |
| T506 | `emotional_text_passages` | Emotional text passages | exploratory | positive, negative, neutral | state |
| T507 | `readability_difficulty_ladder` | Readability difficulty ladder | promising | easy, medium, hard | state |
| T508 | `inner_speech_silent_repetition` | Inner speech vs fixation | exploratory | inner_speech, fixation | none |

### Tier 6 — Audio listening & music

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T601 | `music_speech_noise_3class` | Music vs speech vs noise | promising | music, speech, noise | state |
| T602 | `beat_tempo_tracking` | Beat/tempo tracking (60 vs 120 BPM) | exploratory | tempo_60, tempo_120 | none |
| T603 | `consonant_vs_dissonant_chords` | Consonant vs dissonant chords | exploratory | consonant, dissonant | state |
| T604 | `familiar_vs_unfamiliar_melody` | Familiar vs unfamiliar melody | exploratory | familiar, unfamiliar | none |
| T605 | `narrative_vs_reversed_speech` | Narrative vs reversed speech | exploratory | forward, reversed | state |
| T606 | `nature_vs_urban_soundscapes` | Nature vs urban soundscapes | promising | nature, urban | state |
| T607 | `loudness_annoyance_ladder` | Loudness/annoyance ladder | exploratory | quiet, loud | state |
| T608 | `selective_attention_two_streams` | Selective attention, two streams | exploratory | attend_left, attend_right | state |

### Tier 7 — Imagery, control & closed-loop agent

| id | slug | title | feasibility | classes | agent_use |
|---|---|---|---|---|---|
| T701 | `blink_switch_control` | Blink-controlled switch | established | blink, rest | command |
| T702 | `clench_switch_control` | Jaw-clench switch | established | clench, rest | command |
| T703 | `eyebrow_raise_switch` | Eyebrow-raise switch | promising | raise, rest | command |
| T704 | `motor_imagery_left_right` | Motor imagery left vs right | exploratory | left, right | none |
| T705 | `visual_imagery_vs_percept` | Visual imagery vs perception | exploratory | percept, imagery | none |
| T706 | `closed_loop_blink_confirm_agent` | Closed loop: blink-confirm handshake | established | confirm, cancel | both |
| T707 | `closed_loop_load_adaptive_pacing` | Closed loop: load-adaptive agent pacing | exploratory | load_high, load_low | state |
| T708 | `closed_loop_relaxation_neurofeedback` | Closed loop: relaxation neurofeedback | promising | relax, effort | both |

## Writing rules

1. Write **only** your assigned files, one YAML per test.
2. `duration_min` must respect a **45 minute** session budget: keep individual
   tests at 4–16 min, and note block breaks where the test exceeds 10 min.
3. Class labels are lowercase snake_case and must match `markers` keys and
   `stimuli.queries` keys exactly.
4. `iti_jitter_s` is required for every ERP test; without jitter the analysis
   cannot separate the evoked response from ongoing alpha.
5. `risks` must contain at least one *specific* failure mode for that test
   (not boilerplate). Say what would make it fail on this montage.
6. Where a test needs speech or images that may not be downloadable, set
   `stimuli.fallback` so it still runs (e.g. `synth` for tones/syllables,
   `builtin` for images, `textbank` for text).
7. Validate before finishing:
   `cd /path/to/neurobridge && .venv/bin/python -c "from neurobridge.catalog import validate_catalog; print(validate_catalog())"`
   It must print `{}` for your files.
