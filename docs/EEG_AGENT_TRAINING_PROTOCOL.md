# EEG-to-agent personal calibration protocol

**Version:** 1.0 | **Prepared:** 2026-10-13 | **Deliverable:** protocol and test materials for an AI agent

**Status:** study design and stimulus candidates. No EEG has been collected or decoded for this document. No model, device driver, app, or validated brain translator is included.

## 1. Instructions to the receiving AI agent

You are helping one participant evaluate a personal EEG interface. Use the 64 tests below to collect labeled examples, evaluate reproducible patterns, and design a limited interface to their AI agent. Treat this document as a specification, not as evidence that its proposed labels are decodable.

Your first useful output is a recording plan and an inventory of the participant's raw export format. Then prepare the selected stimuli, record synchronized trials, and evaluate a small classifier. Do not claim to read private thoughts, infer facts about the participant from a stimulus label, or convert uncertain EEG into fluent statements attributed to them.

Maintain three separate fields throughout:

1. `task_label`: what the trial asked the participant to do or presented to them.
2. `reported_label`: what the participant actually reported afterward, including unknown, skipped, or unsuccessful.
3. `predicted_label`: what a model estimated from eligible EEG, with uncertainty and signal-quality information.

A picture of a forest is a forest stimulus. It is not a ground-truth label for calmness. A YES cue is a requested intention. It is not proof of an internally expressed YES. A correct text answer demonstrates task performance, not the neural mechanism used.

Work locally with the participant's recordings unless they explicitly choose another destination. The agent can receive small, structured predictions instead of raw EEG. Keep API credentials out of this file, prompts, exported trial logs, and repositories. This protocol does not authorize messages, purchases, device changes, or other external actions based on predicted commands.

### Recommended initial targets

| Priority | Target | Appropriate interpretation |
|---|---|---|
| 1 | Recording quality and artifact detection | Identify unusable data and possible eye, jaw, or movement contamination. |
| 2 | YES / NO / no command | Test a deliberately expressed, small command vocabulary. Success is uncertain. |
| 3 | Relative task demand | Estimate patterns associated with easy versus harder tasks for this person and setup. |
| 4 | Self-reported preference | Explore associations with the person's actual ratings; do not assign universal emotional labels. |
| 5 | Four presented concepts: water, fire, dog, piano | Explore stimulus-related patterns and cross-modal transfer; do not equate this with general thought decoding. |

If the available EEG cannot discriminate a target on independent sessions, report that result and narrow the interface. Do not fabricate a translation to make the agent appear more capable. Noninvasive language decoding remains experimental and highly dependent on recording device, task, data volume, and evaluation method. [R1]

## 2. Participant and hardware context

| Field | Known information or required action |
|---|---|
| Advertised device name supplied by participant | `TH21A_D937B53E3FCC` |
| Product identification supplied by participant | NingNao / 宁脑 biosensing headband; three physical electrodes |
| Chipset identification supplied by participant | Nordic nRF52; Nordic DFU service `0xFE59` observed |
| Raw EEG access | Participant confirms they can export raw EEG. |
| Independently verified exact model specifications | Not established. Do not infer specifications from the advertised name. |
| EEG channel count | Unknown. Three electrodes do not establish three independent channels; active, reference, and bias/ground electrodes can have different roles. |
| Required before filtering or interpreting EEG | Actual channel layout, electrode locations and roles, sample rate, amplitude units or ADC scaling, reference, firmware/export version, and timestamp/counter behavior. |
| Required before trial alignment | Determine whether exported timestamps describe acquisition or packet arrival; identify missing samples and clock resets. |

Nordic documents `0xFE59` as a firmware-update service. It is not an EEG streaming specification. Do not write to it or enter DFU mode for this protocol. Do not invent EEG characteristic UUIDs, packet layouts, a BrainFlow board ID, or a sampling rate. [R2]

**Hardware gate:** inspect a short raw export and the existing working reader before scheduling the full study. Verify whether this is one or several independent EEG channels. Preserve the original export. If there is only one channel, use per-channel temporal/spectral analysis; do not promise spatial localization, independent-component artifact separation, or reliable motor-imagery control. If sensor locations are frontal, pay particular attention to eye and facial-muscle contamination.

## 3. What the 64 tests mean

The numbered items are **64 distinct test types**, not 64 samples sufficient to train a translator. Each usable repetition is a trial. Pilot the range, then repeat a smaller set relevant to the chosen target.

| Module | IDs | Purpose |
|---|---|---|
| A | T01-T08 | Baseline and artifact calibration |
| B | T09-T16 | Reading and text controls |
| C | T17-T24 | Image observation and personal ratings |
| D | T25-T32 | Listening and personal ratings |
| E | T33-T40 | Agreement of meaning across text, images, and sounds |
| F | T41-T48 | Task demand and memory |
| G | T49-T56 | Intentional commands, rest, and free choice |
| H | T57-T64 | Reserved texts, media, and command cues |

T01-T08 are quality controls. T09-T56 are development tasks. T57-T64 are reserved evaluation tasks; never include their EEG in model fitting, feature selection, or threshold selection. Viewing an asset for source/technical review before recording is allowed, but do not rehearse its task with the participant. Keep the participant-facing held-out stimuli hidden until evaluation.

### Session conditions

Use the same seat, display, viewing distance, headband placement, and comfortable audio setting. Record deviations instead of assuming identical conditions. Use silent reading and a relaxed face. Collect answers and ratings after the EEG analysis interval. Avoid deliberate blinking, speaking, typing, or jaw movement during cognitive trials; ordinary behavior still happens and must be annotated, not punished.

Use comfortable sound levels and static images. No strobing, electrical stimulation, startling volume, breath-holding, or induced distress is part of this protocol. Stop or skip a trial if the participant is uncomfortable. Record sleep, fatigue, caffeine, and time of day as optional session context, not diagnostic labels.

## 4. Shared timing and presentation contract

Use an instrumented presenter connected to the existing reader or recording software. This Markdown file is a specification, not a precision presentation engine. Do not run synchronized trials by manually scrolling through it and claim precise onset times. Manual presentation is suitable for rehearsal; label such recordings `timing_quality: coarse` and restrict analysis to long, buffered windows.

For T09-T64, use this default sequence unless the card overrides it:

| Phase | Timing | Record and display |
|---|---|---|
| Instructions | Self-paced, before analysis | Explain the task; do not reveal the later answer key. |
| Washout | 4 s | Neutral background; remove the instruction/cue. |
| Baseline | 4 s | Same central fixation mark for every condition. |
| Stimulus/task | Duration on card | Record actual start and end; show only the designated stimulus. |
| Post-task gap | 2 s | Fixation; no answer prompt yet. |
| Response | Up to 15 s | Present the question and collect the actual answer, timeout, or skip. |
| Ratings | Up to 15 s | Collect the defined ratings; these are self-reports. |
| Intertrial interval | Random 2-4 s | Fixation; record the realized duration. |

For command trials, show the intention cue during instructions only. After washout and baseline, show identical fixation during the 6 s intention interval. Do not display the command word, use different colors, or play different sounds during that interval.

**Default analysis intervals:** for an 8 s image/audio/multimodal task, start with 1-7 s after verified onset; for 12 s reading, start with 2-10 s; for 6 s intentional-command tasks, start with 2-6 s. These buffered intervals are design choices for sustained activity, not evidence of precise neural onset. Use the full interval for behavioral timing. Set different intervals only before evaluating the locked test set. T45-T46 require their own sequence interval; the probe is outside it.

Preload media before baseline. Display images with preserved aspect ratio inside the same fixed frame on a neutral background. Use the same text style and contrast. Do not show filenames, titles, search queries, source pages, answers, or class labels during acquisition. Attribution belongs in the source ledger and a separate credits view.

All audio cards use an **8 s prepared excerpt**. Select a representative segment with the stated sound audible, log exact source start/end, and save a local copy. If a preview is too short or unsuitable, replace it with a different recording; do not silently loop or pad it. Apply a consistent short fade, inspect clipping, and match approximate loudness without making it uncomfortable. Save the processed file hash and processing settings. Source previews are lossy; record that fact. Play only the prepared local excerpt during a trial.

Image/audio onset estimates need measured display/audio latency for fine timing. Wall-clock timestamps and Bluetooth packet-arrival timestamps are not interchangeable with acquisition time. If using LSL, record the EEG and marker streams together and preserve synchronization information, for example in XDF. LSL does not by itself measure the time photons or sound reach the participant. [R3]

### Common responses and ratings

Every cognitive trial permits `skip`, `not_sure`, and `timed_out`. Use these values rather than forcing a class.

After the card-specific answer, collect:

- `effort_1_5`: 1 very easy; 2 easy; 3 moderate; 4 hard; 5 very hard.
- `pleasantness_1_5`: 1 very unpleasant; 2 unpleasant; 3 neutral; 4 pleasant; 5 very pleasant.
- `activation_1_5`: 1 very low; 2 low; 3 moderate; 4 high; 5 very high.
- `response_confidence_1_5`: 1 guessing; 2 unsure; 3 somewhat sure; 4 sure; 5 very sure.
- `artifact_report`: none noticed, blink, gaze movement, jaw movement, body movement, or other; allow multiple.

These ratings are separate targets. Do not collapse pleasantness, activation, and attention into one emotion score. For command cards also collect `intention_completed: yes/no/not_sure` and the intention actually attempted.

## 5. Media sourcing and preparation

The source ledger at the end contains **20 real candidate records retrieved through the Openverse API on 2026-10-13**: ten image records and ten audio records. The audio records point to files and landing pages on Freesound; Openverse was used to discover their public metadata and preview URLs. Images are from the Openverse image API as requested.

The image compositions were inspected for basic task fit. Audio has **not been auditioned** for this document. All candidates still need original-source license verification; audio also needs excerpt selection and content review. No cached stimuli, processed clips, or verified file hashes are supplied. The agent must mark them `candidate`, then `approved` only after preparation. Do not report that media preparation or EEG validation is already complete.

### Openverse image API

Use `GET https://api.openverse.org/v1/images/` with a query, small page size, and permitted license filters. Example:

```text
https://api.openverse.org/v1/images/?q=flowing%20stream&page_size=4&license=cc0%2Cby&filter_dead=true&mature=false
```

Follow each record's `detail_url` and `foreign_landing_url`. Preserve `id`, title, creator, creator URL, license, license version/URL, source URL, and attribution. A search result is not an assertion that every indexed license is correct. Verify at the original source, and honor rate limits and `Retry-After`; do not assume a fixed rate limit. Registration may be appropriate for larger searches. [R4]

### Freesound audio

The supplied public previews can be reviewed without embedding credentials in this file. For updated Freesound metadata or further discovery, use the current Freesound API and a participant-owned token stored outside the protocol:

```text
GET https://freesound.org/apiv2/search/?query=fire%20crackling&page_size=5&fields=id,name,username,url,license,duration,previews,gen_ai_preference
Authorization: Token <read from secure local configuration>
```

Confirm the current endpoint and parameters against the official API reference before implementation. Token authentication supports many read-only resources; original-file downloads require OAuth2. The ledger points to previews, not an OAuth download grant. [R5, R6]

Store any uploader `gen_ai_preference` with the source record. It was not exposed in the Openverse records retrieved here. Do not silently pass media into generative-model training or cloud models. Review current source preferences if such use is proposed; showing a sound to obtain EEG labels and training a generative model on the sound are different activities. [R6, R7]

### Asset acceptance rules

Prefer CC0 or CC BY, with retained attribution. Check every candidate at its original landing page. Replace deleted, mismatched, unsuitable, or insufficiently licensed material and record the replacement. Avoid prominent speech in nonverbal audio tasks. Exclude graphic or distressing content. Treat similar clips from the same original recording as one source group for data splitting.

Do not search or fetch remote assets during EEG trials. Cache the exact approved image/audio bytes locally, compute SHA-256, and lock the asset ledger for that dataset version. Record edits, resizing, gain changes, crop intervals, and preview/original status. Replacing an asset creates a new `stimulus_id`; do not silently reuse the old identity.

**Holdout lock:** `IMG_DOG_H`, `IMG_PIANO_H`, `AUD_WATER_H`, and `AUD_FIRE_H` are reserved for T59-T62. Do not use them as training examples or substitution assets. T57-T58 contain reserved passages; T63-T64 contain reserved cue wording.

## 6. The 64 test cards

The exact words under **Stimulus** are original supplied text, except for credited media titles. The agent may present only the participant instruction and stimulus during a trial. Keep expected answers, labels, notes, and source metadata in the experiment record, outside the participant's stimulus view. Training passage variants alternate evenly within each recording day; do not use one variant exclusively on one day.

### Module A: Recording quality and artifacts

#### T01. Eyes-open fixation baseline

- **Type / task duration:** `calibration` / 60 s.
- **Participant instruction:** Look gently at the central mark. Sit comfortably with a relaxed face.
- **Stored task label:** `quality_control/eyes_open`.
- **Post-task response:** After offset: Were you awake, comfortable, and able to keep looking at the mark?
- **Expected answer or ground truth:** Self-report only.

**Stimulus:**

> A small static + on a neutral background.

**Protocol note:** Use the middle 50 s for baseline summaries. Log spontaneous blinks; do not discard the entire trial solely because one blink occurs.

#### T02. Eyes-closed baseline

- **Type / task duration:** `calibration` / 45 s.
- **Participant instruction:** Close your eyes gently and stay awake. Open them only after the end signal.
- **Stored task label:** `quality_control/eyes_closed`.
- **Post-task response:** After opening your eyes: Were you awake throughout?
- **Expected answer or ground truth:** Self-report only.

**Stimulus:**

> No changing visual stimulus. Give the end signal after the recorded interval.

**Protocol note:** Log actual closure/opening if available; exclude 5 s at each edge. A spectral difference is not a universal relaxation score.

#### T03. Blink artifact examples

- **Type / task duration:** `calibration` / 30 s.
- **Participant instruction:** At each cue, blink once naturally, then relax.
- **Stored task label:** `artifact_calibration/blink`.
- **Post-task response:** How many requested blinks did you complete? Did any others occur?
- **Expected answer or ground truth:** Requested count 5; actual completion is reported.

**Stimulus:**

> Five small cue markers at 5, 10, 15, 20, and 25 s.

**Protocol note:** Record cue times separately from observed blink times. Use these to characterize contamination, not to train a cognitive command.

#### T04. Gaze-shift artifact examples

- **Type / task duration:** `calibration` / 30 s.
- **Participant instruction:** At each cue, briefly look left, return to center, then look right and return. Keep your head still.
- **Stored task label:** `artifact_calibration/gaze`.
- **Post-task response:** Did you complete the gaze shifts? Did your head move?
- **Expected answer or ground truth:** Self-report or observation.

**Stimulus:**

> Five identical cue markers at 5, 10, 15, 20, and 25 s.

**Protocol note:** Annotate the entire movement interval; eye movements can dominate a headband recording.

#### T05. Jaw-muscle artifact examples

- **Type / task duration:** `calibration` / 30 s.
- **Participant instruction:** At each cue, gently press your teeth together for about one second, then fully relax. Skip if uncomfortable.
- **Stored task label:** `artifact_calibration/jaw`.
- **Post-task response:** How many gentle jaw actions did you complete?
- **Expected answer or ground truth:** Requested count 5; actual completion is reported.

**Stimulus:**

> Five identical cue markers at 5, 10, 15, 20, and 25 s.

**Protocol note:** No forceful clenching. Treat increased high-frequency activity as possible muscle contamination, not evidence of concentration.

#### T06. Actual left-finger movement

- **Type / task duration:** `calibration` / 30 s.
- **Participant instruction:** At each cue, lightly tap one left-hand finger once; keep your face relaxed.
- **Stored task label:** `artifact_calibration/left_movement`.
- **Post-task response:** How many taps did you complete?
- **Expected answer or ground truth:** Requested count 5; actual completion is reported.

**Stimulus:**

> Five identical cue markers at 5, 10, 15, 20, and 25 s.

**Protocol note:** This is actual movement, not motor imagery. A signal difference may be movement or electrode artifact.

#### T07. Actual right-finger movement

- **Type / task duration:** `calibration` / 30 s.
- **Participant instruction:** At each cue, lightly tap one right-hand finger once; keep your face relaxed.
- **Stored task label:** `artifact_calibration/right_movement`.
- **Post-task response:** How many taps did you complete?
- **Expected answer or ground truth:** Requested count 5; actual completion is reported.

**Stimulus:**

> Five identical cue markers at 5, 10, 15, 20, and 25 s.

**Protocol note:** Match T06 cues and posture; preserve all movement annotations.

#### T08. Unstructured quiet baseline

- **Type / task duration:** `calibration` / 60 s.
- **Participant instruction:** Rest with your eyes open. Let thoughts come and go without performing a special mental task.
- **Stored task label:** `quality_control/quiet_rest`.
- **Post-task response:** After offset: Describe your state as quiet, distracted, sleepy, or unsure.
- **Expected answer or ground truth:** Self-report only.

**Stimulus:**

> The same static + as T01.

**Protocol note:** This is a comparison with instructed fixation, not a known empty mind. Repeat a short quiet baseline at session boundaries.

### Module B: Reading and text controls

#### T09. Read about water

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once, then keep the scene in mind until the text disappears.
- **Stored task label:** `presented_concept/water`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** water

**Stimulus:**

> Variant A: A narrow stream passes over smooth stones. Clear water bends around a fallen branch, forms a small pool, and continues toward the shaded bank.

> Variant B: Water flows through a shallow channel beside the path. It divides around three rocks, joins again, and moves steadily beyond a low wooden bridge.

**Protocol note:** Alternate passages evenly within days. The answer checks comprehension; the stimulus label is not an inferred thought.

#### T10. Read about fire

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once, then keep the scene in mind until the text disappears.
- **Stored task label:** `presented_concept/fire`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** fire

**Stimulus:**

> Variant A: A small fire burns inside a stone fireplace. Orange flames rise around a short log, and a few glowing embers settle beneath the grate.

> Variant B: The fireplace holds two dry logs above a bed of embers. A flame briefly leans to one side, then rises while the wood slowly darkens.

#### T11. Read about a dog

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once, then keep the scene in mind until the text disappears.
- **Stored task label:** `presented_concept/dog`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** dog

**Stimulus:**

> Variant A: A dog waits beside a closed garden gate. It turns toward a familiar voice, lifts its ears, and walks back to the person holding its lead.

> Variant B: The dog carries a worn ball across the lawn. It places the ball near a chair, steps back, and watches the person seated in the shade.

#### T12. Read about a piano

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once, then keep the scene in mind until the text disappears.
- **Stored task label:** `presented_concept/piano`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** piano

**Stimulus:**

> Variant A: A piano stands beside the window in a quiet room. The player lifts the lid, sits on the bench, and presses three nearby keys in sequence.

> Variant B: The pianist opens a thin book on the music stand. One hand rests above the white keys while the other begins a short, evenly spaced melody.

#### T13. Literal text comprehension

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently and remember the sequence of events.
- **Stored task label:** `reading_task/literal_comprehension`.
- **Post-task response:** Where is the notebook at the end?
- **Expected answer or ground truth:** upper drawer

**Stimulus:**

> Nora puts a blue notebook on the desk and a red folder on the chair. Before leaving, she moves the notebook into the upper drawer.

**Protocol note:** Do not reveal this question or the answer during the reading interval.

#### T14. Simple text inference

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently and consider what probably happened.
- **Stored task label:** `reading_task/inference`.
- **Post-task response:** What most likely happened to the ice cube?
- **Expected answer or ground truth:** It melted.

**Stimulus:**

> Mara leaves an ice cube in a shallow bowl on the kitchen table. An hour later the bowl contains a small pool of water and no solid ice.

**Protocol note:** Accept equivalent wording. This is an inference task; its demand must be measured by performance and self-report.

#### T15. Text with negation

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently, paying attention to where the object actually is.
- **Stored task label:** `reading_task/negation`.
- **Post-task response:** Where are the keys?
- **Expected answer or ground truth:** blue box beside the lamp

**Stimulus:**

> The keys are not inside the drawer, and they are not under the chair. They are in the blue box beside the lamp.

**Protocol note:** Store the actual answer rather than treating every completed trial as successful comprehension.

#### T16. Reading command words control

- **Type / task duration:** `sequence` / 8 s.
- **Participant instruction:** Read each word silently as ordinary text. Do not deliberately issue a command to the agent.
- **Stored task label:** `control/command_word_reading`.
- **Post-task response:** Were you just reading, or did you also deliberately intend a command?
- **Expected answer or ground truth:** No objective answer; record reading_only, command_also, or not_sure.

**Stimulus:**

> Show YES, NO, YES, NO centrally, one word for 2 s each, with identical font and background.

**Protocol note:** Exclude command_also trials from the clean reading control. This helps expose cue or word-reading contamination of command decoding.

### Module C: Images

#### T17. Observe an image: water

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally without speaking or deliberately issuing a command.
- **Stored task label:** `presented_concept/water`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** water
- **Assets:** `IMG_WATER_A`.

**Stimulus:**

> Show IMG_WATER_A only.

**Protocol note:** Record the identification answer and common ratings. Image category alone does not establish internal meaning or preference.

#### T18. Observe an image: fire

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally without speaking or deliberately issuing a command.
- **Stored task label:** `presented_concept/fire`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** fire
- **Assets:** `IMG_FIRE_A`.

**Stimulus:**

> Show IMG_FIRE_A only.

**Protocol note:** Record the identification answer and common ratings. Image category alone does not establish internal meaning or preference.

#### T19. Observe an image: dog

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally without speaking or deliberately issuing a command.
- **Stored task label:** `presented_concept/dog`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** dog
- **Assets:** `IMG_DOG_A`.

**Stimulus:**

> Show IMG_DOG_A only.

**Protocol note:** Record the identification answer and common ratings. Image category alone does not establish internal meaning or preference.

#### T20. Observe an image: piano

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally without speaking or deliberately issuing a command.
- **Stored task label:** `presented_concept/piano`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** piano
- **Assets:** `IMG_PIANO_A`.

**Stimulus:**

> Show IMG_PIANO_A only.

**Protocol note:** Record the identification answer and common ratings. Image category alone does not establish internal meaning or preference.

#### T21. Observe and rate: forest path

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally and notice your own response.
- **Stored task label:** `preference_stimulus/forest`.
- **Post-task response:** Would you choose to see this image again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `IMG_FOREST_A`.

**Stimulus:**

> Show IMG_FOREST_A only.

**Protocol note:** Use the actual pleasantness, activation, and choice ratings as labels. Do not code nature as calm or tools as interesting by default.

#### T22. Observe and rate: city street

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally and notice your own response.
- **Stored task label:** `preference_stimulus/street`.
- **Post-task response:** Would you choose to see this image again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `IMG_STREET_A`.

**Stimulus:**

> Show IMG_STREET_A only.

**Protocol note:** Use the actual pleasantness, activation, and choice ratings as labels. Do not code nature as calm or tools as interesting by default.

#### T23. Observe and rate: flowers

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally and notice your own response.
- **Stored task label:** `preference_stimulus/flowers`.
- **Post-task response:** Would you choose to see this image again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `IMG_FLOWERS_A`.

**Stimulus:**

> Show IMG_FLOWERS_A only.

**Protocol note:** Use the actual pleasantness, activation, and choice ratings as labels. Do not code nature as calm or tools as interesting by default.

#### T24. Observe and rate: hand tools

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at the image naturally and notice your own response.
- **Stored task label:** `preference_stimulus/tools`.
- **Post-task response:** Would you choose to see this image again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `IMG_TOOLS_A`.

**Stimulus:**

> Show IMG_TOOLS_A only.

**Protocol note:** Use the actual pleasantness, activation, and choice ratings as labels. Do not code nature as calm or tools as interesting by default.

### Module D: Freesound listening

#### T25. Listen: water

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Do not speak or move to the sound.
- **Stored task label:** `presented_concept/water`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** water
- **Assets:** `AUD_WATER_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_WATER_A.

**Protocol note:** The candidate must be auditioned and cropped first. Record actual playback onset, excerpt boundaries, gain, and local file hash.

#### T26. Listen: fire

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Do not speak or move to the sound.
- **Stored task label:** `presented_concept/fire`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** fire
- **Assets:** `AUD_FIRE_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_FIRE_A.

**Protocol note:** The candidate must be auditioned and cropped first. Record actual playback onset, excerpt boundaries, gain, and local file hash.

#### T27. Listen: dog

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Do not speak or move to the sound.
- **Stored task label:** `presented_concept/dog`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** dog
- **Assets:** `AUD_DOG_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_DOG_A.

**Protocol note:** The candidate must be auditioned and cropped first. Record actual playback onset, excerpt boundaries, gain, and local file hash.

#### T28. Listen: piano

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Do not speak or move to the sound.
- **Stored task label:** `presented_concept/piano`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** piano
- **Assets:** `AUD_PIANO_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_PIANO_A.

**Protocol note:** The candidate must be auditioned and cropped first. Record actual playback onset, excerpt boundaries, gain, and local file hash.

#### T29. Listen and rate: birds

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Notice your own reaction.
- **Stored task label:** `preference_stimulus/birds`.
- **Post-task response:** Would you choose to hear this sound again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `AUD_BIRDS_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_BIRDS_A.

**Protocol note:** Use reported ratings rather than assuming birds are pleasant or traffic is stressful. Replace speech-dominated excerpts.

#### T30. Listen and rate: traffic

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Notice your own reaction.
- **Stored task label:** `preference_stimulus/traffic`.
- **Post-task response:** Would you choose to hear this sound again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `AUD_TRAFFIC_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_TRAFFIC_A.

**Protocol note:** Use reported ratings rather than assuming birds are pleasant or traffic is stressful. Replace speech-dominated excerpts.

#### T31. Listen and rate: rain

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Notice your own reaction.
- **Stored task label:** `preference_stimulus/rain`.
- **Post-task response:** Would you choose to hear this sound again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `AUD_RAIN_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_RAIN_A.

**Protocol note:** Use reported ratings rather than assuming birds are pleasant or traffic is stressful. Replace speech-dominated excerpts.

#### T32. Listen and rate: machine hum

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen while looking at the neutral fixation mark. Notice your own reaction.
- **Stored task label:** `preference_stimulus/machine_hum`.
- **Post-task response:** Would you choose to hear this sound again: yes, no, or unsure?
- **Expected answer or ground truth:** Participant report; no preassigned preference.
- **Assets:** `AUD_MACHINE_A`.

**Stimulus:**

> Play the approved 8 s excerpt from AUD_MACHINE_A.

**Protocol note:** Use reported ratings rather than assuming birds are pleasant or traffic is stressful. Replace speech-dominated excerpts.

### Module E: Meaning across modalities

#### T33. Text-image agreement: water / water

- **Type / task duration:** `text_image` / 8 s.
- **Participant instruction:** Read the sentence and inspect the image. Decide whether their main topics agree.
- **Stored task label:** `cross_modal_agreement/match`.
- **Post-task response:** Do the main topics match: yes, no, or unsure?
- **Expected answer or ground truth:** yes
- **Assets:** `IMG_WATER_A`.

**Stimulus:**

> Display IMG_WATER_A and this sentence together: A stream of water moves around the stones.

**Protocol note:** Show the sentence in the same position on all four cards. These repeated pairs can be memorized; success alone does not demonstrate general semantic decoding.

#### T34. Text-image agreement: water / fire

- **Type / task duration:** `text_image` / 8 s.
- **Participant instruction:** Read the sentence and inspect the image. Decide whether their main topics agree.
- **Stored task label:** `cross_modal_agreement/mismatch`.
- **Post-task response:** Do the main topics match: yes, no, or unsure?
- **Expected answer or ground truth:** no
- **Assets:** `IMG_FIRE_A`.

**Stimulus:**

> Display IMG_FIRE_A and this sentence together: A stream of water moves around the stones.

**Protocol note:** Show the sentence in the same position on all four cards. These repeated pairs can be memorized; success alone does not demonstrate general semantic decoding.

#### T35. Text-image agreement: fire / fire

- **Type / task duration:** `text_image` / 8 s.
- **Participant instruction:** Read the sentence and inspect the image. Decide whether their main topics agree.
- **Stored task label:** `cross_modal_agreement/match`.
- **Post-task response:** Do the main topics match: yes, no, or unsure?
- **Expected answer or ground truth:** yes
- **Assets:** `IMG_FIRE_A`.

**Stimulus:**

> Display IMG_FIRE_A and this sentence together: A small fire burns around the logs.

**Protocol note:** Show the sentence in the same position on all four cards. These repeated pairs can be memorized; success alone does not demonstrate general semantic decoding.

#### T36. Text-image agreement: fire / water

- **Type / task duration:** `text_image` / 8 s.
- **Participant instruction:** Read the sentence and inspect the image. Decide whether their main topics agree.
- **Stored task label:** `cross_modal_agreement/mismatch`.
- **Post-task response:** Do the main topics match: yes, no, or unsure?
- **Expected answer or ground truth:** no
- **Assets:** `IMG_WATER_A`.

**Stimulus:**

> Display IMG_WATER_A and this sentence together: A small fire burns around the logs.

**Protocol note:** Show the sentence in the same position on all four cards. These repeated pairs can be memorized; success alone does not demonstrate general semantic decoding.

#### T37. Image-sound agreement: dog / dog

- **Type / task duration:** `image_audio` / 8 s.
- **Participant instruction:** Look and listen. Decide whether the pictured object fits the main sound.
- **Stored task label:** `cross_modal_agreement/match`.
- **Post-task response:** Do the image and main sound match: yes, no, or unsure?
- **Expected answer or ground truth:** yes
- **Assets:** `IMG_DOG_A`, `AUD_DOG_A`.

**Stimulus:**

> Display IMG_DOG_A while playing the approved 8 s excerpt from AUD_DOG_A.

**Protocol note:** Log image and audio onsets separately and their difference. A failed or delayed component invalidates the paired trial.

#### T38. Image-sound agreement: dog / piano

- **Type / task duration:** `image_audio` / 8 s.
- **Participant instruction:** Look and listen. Decide whether the pictured object fits the main sound.
- **Stored task label:** `cross_modal_agreement/mismatch`.
- **Post-task response:** Do the image and main sound match: yes, no, or unsure?
- **Expected answer or ground truth:** no
- **Assets:** `IMG_DOG_A`, `AUD_PIANO_A`.

**Stimulus:**

> Display IMG_DOG_A while playing the approved 8 s excerpt from AUD_PIANO_A.

**Protocol note:** Log image and audio onsets separately and their difference. A failed or delayed component invalidates the paired trial.

#### T39. Image-sound agreement: piano / piano

- **Type / task duration:** `image_audio` / 8 s.
- **Participant instruction:** Look and listen. Decide whether the pictured object fits the main sound.
- **Stored task label:** `cross_modal_agreement/match`.
- **Post-task response:** Do the image and main sound match: yes, no, or unsure?
- **Expected answer or ground truth:** yes
- **Assets:** `IMG_PIANO_A`, `AUD_PIANO_A`.

**Stimulus:**

> Display IMG_PIANO_A while playing the approved 8 s excerpt from AUD_PIANO_A.

**Protocol note:** Log image and audio onsets separately and their difference. A failed or delayed component invalidates the paired trial.

#### T40. Image-sound agreement: piano / dog

- **Type / task duration:** `image_audio` / 8 s.
- **Participant instruction:** Look and listen. Decide whether the pictured object fits the main sound.
- **Stored task label:** `cross_modal_agreement/mismatch`.
- **Post-task response:** Do the image and main sound match: yes, no, or unsure?
- **Expected answer or ground truth:** no
- **Assets:** `IMG_PIANO_A`, `AUD_DOG_A`.

**Stimulus:**

> Display IMG_PIANO_A while playing the approved 8 s excerpt from AUD_DOG_A.

**Protocol note:** Log image and audio onsets separately and their difference. A failed or delayed component invalidates the paired trial.

### Module F: Task demand and memory

#### T41. Single-step mental arithmetic

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Calculate silently; keep the answer in mind until the response prompt.
- **Stored task label:** `task_demand/arithmetic_single_step`.
- **Post-task response:** What is the answer?
- **Expected answer or ground truth:** 15

**Stimulus:**

> Add 7 and 8.

**Protocol note:** For repeat trials, use fresh one-digit additions. Precompute and lock answers. Pair session counts with T42; compare effort and accuracy, not just instructed difficulty.

#### T42. Two-step mental arithmetic

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Calculate silently; keep the answer in mind until the response prompt.
- **Stored task label:** `task_demand/arithmetic_two_step`.
- **Post-task response:** What is the answer?
- **Expected answer or ground truth:** 83

**Stimulus:**

> Multiply 17 by 6, then subtract 19.

**Protocol note:** For repeat trials, use fresh two-digit times one-digit problems followed by subtraction. Precompute answers. Adapt difficulty during pilot only, then freeze the rule.

#### T43. Retain three digits

- **Type / task duration:** `sequence` / 10 s.
- **Participant instruction:** Remember the digits in order without speaking or moving your lips.
- **Stored task label:** `task_demand/digit_span_3`.
- **Post-task response:** Type the remembered digits after the response prompt.
- **Expected answer or ground truth:** 583

**Stimulus:**

> Show 5 8 3 centrally for 3 s, then fixation for 7 s.

**Protocol note:** Analyze retention separately from visual encoding. Generate fresh three-digit sequences for repeats; allow leading zeros and store them as strings.

#### T44. Retain six digits

- **Type / task duration:** `sequence` / 10 s.
- **Participant instruction:** Remember the digits in order without speaking or moving your lips.
- **Stored task label:** `task_demand/digit_span_6`.
- **Post-task response:** Type the remembered digits after the response prompt.
- **Expected answer or ground truth:** 742916

**Stimulus:**

> Show 7 4 2 9 1 6 centrally for 3 s, then fixation for 7 s.

**Protocol note:** Match T43 presentation and retention timing. Use fresh sequences for repeats. Longer span is an intended manipulation; report actual effort and accuracy.

#### T45. One-back comparison

- **Type / task duration:** `sequence` / 13.5 s.
- **Participant instruction:** Count how many letters equal the immediately previous letter. Count silently; do not press a key during the sequence.
- **Stored task label:** `task_demand/one_back`.
- **Post-task response:** How many one-back matches occurred?
- **Expected answer or ground truth:** 2: positions 4 and 6 (positions start at 1).

**Stimulus:**

> Show A, B, A, A, B, B, A, B, A centrally, one letter per 1.5 s, with no gap.

**Protocol note:** For repeats generate fresh 9-letter sequences from A-D; precompute all matches. Use matched sequence pools in T45/T46, balanced across tasks/days. Do not let participants memorize one constant answer.

#### T46. Two-back comparison

- **Type / task duration:** `sequence` / 13.5 s.
- **Participant instruction:** Count how many letters equal the letter two positions earlier. Count silently; do not press a key during the sequence.
- **Stored task label:** `task_demand/two_back`.
- **Post-task response:** How many two-back matches occurred?
- **Expected answer or ground truth:** 3: positions 3, 8, and 9 (positions start at 1).

**Stimulus:**

> Show A, B, A, A, B, B, A, B, A centrally, one letter per 1.5 s, with no gap.

**Protocol note:** Use the same sequence-generation rules as T45 and precompute answers. Do not treat the first two positions as eligible matches. Log every letter onset.

#### T47. Apply one decision rule

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Solve the rule silently and retain the answer.
- **Stored task label:** `task_demand/one_rule`.
- **Post-task response:** Which word should you choose?
- **Expected answer or ground truth:** RED

**Stimulus:**

> If the shape is a triangle, choose RED. Otherwise choose BLUE. The shape is a triangle.

**Protocol note:** This is a text rule, not a flashing color task. For repeats vary the shape and locked answer while keeping the rule simple.

#### T48. Apply combined decision rules

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Solve the rules silently and retain the answer.
- **Stored task label:** `task_demand/combined_rules`.
- **Post-task response:** Which word should you choose?
- **Expected answer or ground truth:** AMBER

**Stimulus:**

> If the shape is a circle AND the number is even, choose GREEN. If exactly one condition is true, choose AMBER. If neither is true, choose BLUE. Shape: circle. Number: 7.

**Protocol note:** For repeats vary shapes and integers and precompute answers. Compare actual effort/performance with T47; differing text lengths remain a potential confound.

### Module G: Intentional commands

#### T49. Intend YES

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/yes`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention yes; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently intend YES.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T50. Intend NO

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/no`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention no; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently intend NO.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T51. Intend acceptance with alternate wording

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/yes`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention yes; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently choose to accept. Use your agreed YES intention.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T52. Intend refusal with alternate wording

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/no`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention no; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently choose to decline. Use your agreed NO intention.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T53. Intend MORE

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/more`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention more; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently intend MORE: continue or give more detail.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T54. Intend PAUSE

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout. During the intention interval, keep eyes on fixation and your face relaxed. Use the same chosen mental strategy each time.
- **Stored task label:** `requested_intention/pause`.
- **Post-task response:** Which intention did you actually attempt? Did you complete it without deliberate movement?
- **Expected answer or ground truth:** Requested intention pause; use the actual report as the label.

**Stimulus:**

> Instruction-phase cue: During the next intention interval, silently intend PAUSE: stop presenting and wait.

> Intention interval: identical static + only; no command text, audio, or distinct color.

**Protocol note:** Record the participant's strategy before the session. No nods, jaw actions, or covert key presses. Unknown or unsuccessful attempts are not clean class examples. A cue-linked classifier is not yet a free-intention decoder.

#### T55. No-command control

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** During the next interval, rest naturally. Do not deliberately issue a command.
- **Stored task label:** `requested_intention/no_command`.
- **Post-task response:** Did you intentionally issue a command? If so, which one?
- **Expected answer or ground truth:** Use actual report; intended label is no_command.

**Stimulus:**

> Instruction-phase cue: Rest without a deliberate command.

> Task interval: the same static + used for T49-T54.

**Protocol note:** Thoughts occurring naturally do not invalidate rest. This class is essential: do not force every EEG window into a command.

#### T56. Free choice without a class cue

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Before the next interval, privately choose YES, NO, MORE, PAUSE, or no command. During fixation, attempt only that choice. Report it afterward.
- **Stored task label:** `free_choice_intention/participant_report`.
- **Post-task response:** Which choice did you privately select, and did you attempt it successfully?
- **Expected answer or ground truth:** Participant report only; the agent must not know the choice before prediction.

**Stimulus:**

> No choice-specific text or sound. The intention interval contains the same static + as other command cards.

**Protocol note:** Generate a prediction before revealing the answer. Keep these trials separate from cued trials. Choice frequencies may be imbalanced; do not report accuracy without class counts.

### Module H: Reserved evaluation

#### T57. Held-out reading: water

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once and retain the scene.
- **Stored task label:** `heldout_presented_concept/water`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** water

**Stimulus:**

> Below the footbridge, a current turns a leaf in a slow circle. The leaf reaches the edge of the pool, then drifts through a gap between two stones.

**Protocol note:** Reserved passage. Do not use its EEG or wording in training. Report this as a water-reading holdout only.

#### T58. Held-out reading: fire

- **Type / task duration:** `text` / 12 s.
- **Participant instruction:** Read silently once and retain the scene.
- **Stored task label:** `heldout_presented_concept/fire`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** fire

**Stimulus:**

> A log breaks apart above the grate. Small embers fall into the ash while a thin flame rises from the remaining wood and briefly lights the back of the hearth.

**Protocol note:** Reserved passage. Do not use its EEG or wording in training. Report this as a fire-reading holdout only.

#### T59. Held-out image: dog

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at this new image naturally without issuing a command.
- **Stored task label:** `heldout_presented_concept/dog`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** dog
- **Assets:** `IMG_DOG_H`.

**Stimulus:**

> Show IMG_DOG_H only.

**Protocol note:** Reserved source image. Record on a later session, with the model already frozen. A new asset and a new day test different forms of transfer.

#### T60. Held-out image: piano

- **Type / task duration:** `image` / 8 s.
- **Participant instruction:** Look at this new image naturally without issuing a command.
- **Stored task label:** `heldout_presented_concept/piano`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** piano
- **Assets:** `IMG_PIANO_H`.

**Stimulus:**

> Show IMG_PIANO_H only.

**Protocol note:** Reserved source image. Record on a later session, with the model already frozen. A new asset and a new day test different forms of transfer.

#### T61. Held-out sound: water

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen to this new sound while looking at fixation.
- **Stored task label:** `heldout_presented_concept/water`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** water
- **Assets:** `AUD_WATER_H`.

**Stimulus:**

> Play the prepared 8 s excerpt from AUD_WATER_H.

**Protocol note:** Reserved recording. Keep its source group out of training, even if another preview or crop is available. Audition and crop before evaluation, without participant rehearsal.

#### T62. Held-out sound: fire

- **Type / task duration:** `audio` / 8 s.
- **Participant instruction:** Listen to this new sound while looking at fixation.
- **Stored task label:** `heldout_presented_concept/fire`.
- **Post-task response:** Which topic best fits: water, fire, dog, piano, or other?
- **Expected answer or ground truth:** fire
- **Assets:** `AUD_FIRE_H`.

**Stimulus:**

> Play the prepared 8 s excerpt from AUD_FIRE_H.

**Protocol note:** Reserved recording. Keep its source group out of training, even if another preview or crop is available. Audition and crop before evaluation, without participant rehearsal.

#### T63. Held-out intention cue: YES

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout; then attempt the agreed intention during fixation without movement.
- **Stored task label:** `heldout_requested_intention/yes`.
- **Post-task response:** Which intention did you actually attempt, and was the attempt successful?
- **Expected answer or ground truth:** Requested yes; label from actual report.

**Stimulus:**

> Reserved instruction-phase cue: Silently indicate that you agree to proceed.

> Intention interval: the same static + used for T49-T56.

**Protocol note:** Freeze the classifier and thresholds first. These remain cued trials. Evaluate uncued free choice and idle periods separately before claiming voluntary control.

#### T64. Held-out intention cue: NO

- **Type / task duration:** `internal` / 6 s.
- **Participant instruction:** Read the cue before washout; then attempt the agreed intention during fixation without movement.
- **Stored task label:** `heldout_requested_intention/no`.
- **Post-task response:** Which intention did you actually attempt, and was the attempt successful?
- **Expected answer or ground truth:** Requested no; label from actual report.

**Stimulus:**

> Reserved instruction-phase cue: Silently indicate that you do not agree to proceed.

> Intention interval: the same static + used for T49-T56.

**Protocol note:** Freeze the classifier and thresholds first. These remain cued trials. Evaluate uncued free choice and idle periods separately before claiming voluntary control.

## 7. Repetition, randomization, and progression

The following is a **starting collection budget**, not a statistical power calculation or a promise of sufficient training data. At roughly 16-24 trials per session, spread it across several weeks according to comfort. Start with a short pilot and reduce scope if recording quality or task performance is poor.

| Phase | Items | Initial repetition plan | Planned trials |
|---|---|---|---:|
| Hardware/artifact check | T01-T08 | Once each; repeat a failed check after correcting setup | 8 |
| Broad discovery | T09-T48 | Four trials per item, spread across at least four days | 160 |
| Cued intention discovery | T49-T55 | Twenty trials per item, spread across at least four days | 140 |
| Uncued-choice exploration | T56 | Twenty trials, spread across those days | 20 |
| Locked new-material evaluation | T57-T64 | Eight trials per item, split across at least two later days | 64 |
| Locked idle-command evaluation | T55, now assigned to test sessions | Sixteen additional no-command trials on later days | 16 |
| **Total initial plan** | **64 distinct test types** | Additional daily baselines and repeats are outside this total | **408** |

Pilot recordings may enter the development dataset only if the final protocol and preprocessing rules were already in force. If instructions, stimuli, or labeling rules changed, keep the pilot separate. Never turn a failed held-out evaluation into training while still calling the same recordings a holdout.

For four development days, a simple split is days 1-3 for fitting and grouped cross-validation, day 4 for validation and threshold selection. Reserve the two later days for final testing. With so few independent days, confidence about generalization will be limited regardless of trial count. Add further development days if the model or thresholds are unstable; do not solve this by splitting adjacent windows randomly. [R8]

Within each day, distribute classes across early and late session positions. Mix tasks within compatible presentation blocks; do not collect all YES trials first and all NO trials later. Use a saved random seed, equal planned counts for paired comparisons, and no more than three identical target labels consecutively where feasible. Do not imply that repeating a deterministic text creates a new independent semantic example.

Use separate short blocks for text, images, and audio if setup requires it, but balance their order across days. Mix YES, NO, MORE, PAUSE, and rest within command blocks. For primary YES/NO/no-command analyses, T49+T51 and T50+T52 provide more trials than T55; collect extra development T55 trials or use training-only class weighting. Report the actual counts rather than silently presenting an imbalanced set as balanced.

Take a 2-3 minute break after approximately 12 trials, or sooner when needed. Repeat a 60 s eyes-open baseline after repositioning the headband. A session ends when fatigue, discomfort, repeated artifact, or signal loss makes the labels unreliable.

### Advancement gates

1. **Acquisition:** raw samples, units/scaling, channel roles, and dropouts are understood. Otherwise stop model interpretation and fix the recording path.
2. **Alignment:** events can be mapped to acquisition time or sample indices with a stated uncertainty. Coarse timing permits only coarse, buffered analyses.
3. **Data quality:** a predefined usable interval exists for each retained trial. Exclusions and their class/session counts are reported.
4. **Development:** a simple model beats relevant baselines on independent development days without label leakage.
5. **Locked evaluation:** the frozen model, features, normalization, and decision thresholds are tested on later days. Report new-material, new-day, and free-choice results separately.
6. **Agent integration:** expose only targets that pass the user's pre-agreed performance requirements. Otherwise keep the output exploratory or `uncertain`.

Do not choose a universal success threshold such as 80% from this document. Before final testing, agree on acceptable class errors, abstention coverage, and idle false activations for the intended use. A tiny test set cannot establish reliability for important actions. Confidence intervals, per-day results, and the number of independent trials matter.

## 8. Recording and annotation contract

Keep raw EEG immutable, separate from cleaned arrays and annotations. Store the export in its original format and document any conversion. Use participant/session pseudonyms in analysis files; the Bluetooth name is hardware context, not a model feature.

### 8.1 Session metadata template

Fill unknown fields from the reader/export; `null` means unknown, not zero. This is a template, not a device specification.

```json
{
  "schema_version": "1.0",
  "participant_id": "local_participant_01",
  "session_id": "REPLACE_WITH_UNIQUE_SESSION_ID",
  "session_role": "development_train",
  "device_family_reported": "NingNao three-electrode headband",
  "independent_eeg_channel_count": null,
  "channel_names": [],
  "electrode_locations_and_roles": {},
  "sample_rate_hz": null,
  "amplitude_unit": null,
  "adc_to_microvolt_scale": null,
  "reference_description": null,
  "device_firmware_version": null,
  "reader_software_version": null,
  "timestamp_basis": null,
  "device_counter_available": null,
  "clock_mapping_method": null,
  "onset_timing_uncertainty_ms": null,
  "raw_export_path": null,
  "raw_export_sha256": null,
  "protocol_version": "1.0",
  "stimulus_ledger_version": null,
  "randomization_seed": null
}
```

`session_role` is one of `pilot`, `development_train`, `development_validation`, or `locked_test`. Assign roles before model fitting. Repeated sessions from the same day belong to the same day group for splitting unless independence has been justified.

For converted sample data, retain `sample_index`, original device timestamp/counter when present, estimated acquisition time in its named clock domain, host receive time when available, all raw EEG channel values, and packet/dropout indicators. Keep packet-level metadata separately if the export batches samples. Do not reconstruct a uniformly spaced timeline across unmarked missing samples.

### 8.2 Trial record template

```json
{
  "trial_id": "REPLACE_WITH_UNIQUE_TRIAL_ID",
  "session_id": "REPLACE_WITH_SESSION_ID",
  "test_id": "T49",
  "repetition": 1,
  "stimulus_id": "CMD_YES_CUE_A",
  "stimulus_source_group": "command_cue_family_a",
  "task_label": "requested_intention/yes",
  "reported_label": null,
  "predicted_label": null,
  "stimulus_sha256": null,
  "requested_duration_s": 6,
  "actual_duration_s": null,
  "baseline_start_sample": null,
  "stimulus_start_sample": null,
  "stimulus_end_sample_exclusive": null,
  "analysis_start_sample": null,
  "analysis_end_sample_exclusive": null,
  "timing_quality": "unverified",
  "response": null,
  "response_latency_s": null,
  "task_correct": null,
  "intention_completed": null,
  "effort_1_5": null,
  "pleasantness_1_5": null,
  "activation_1_5": null,
  "response_confidence_1_5": null,
  "artifact_report": [],
  "observed_artifacts": [],
  "quality_status": "unreviewed",
  "exclusion_reason": null,
  "dataset_role": "development_train"
}
```

For multimodal trials, replace a single stimulus ID/hash with a component list; retain per-component onset/offset, source group, hashes, and processing records. For sequences, store the exact token sequence and each token's onset/offset. For audio, also store original source ID, crop start/end seconds, playback gain, processing settings, decoded duration, and preview/original status.

### 8.3 Marker contract

At minimum, record `session_start`, `trial_start`, `instruction_off`, `baseline_on`, `baseline_off`, `stimulus_on`, `stimulus_off`, `response_on`, `response`, `trial_end`, and `session_end`. Use additional component and sequence markers where applicable. Mark failures, retries, pauses, and visibility loss explicitly. Emit both start and end for each analysis interval.

Each marker carries a unique `event_id`, `trial_id`, `test_id`, event name, named clock-domain timestamp, host monotonic timestamp when available, and corresponding acquisition sample index when mapped. Store intended timing separately from observed timing. Preserve mapping uncertainty. Do not relabel a failed playback as a valid exposure just because its scheduled start marker was emitted.

Software markers timestamped at a local bridge can carry transport delay. Measure that delay and jitter; do not call server receipt the exact visual or acoustic onset. If an alignment method is missing, write `timing_quality: coarse` or `unverified` rather than inventing synchronized samples.

### 8.4 Label policy

| Target | Label source | Important restriction |
|---|---|---|
| Presented concept | Approved stimulus identity; participant answer stored separately | This is exposure classification, not proof of imagined meaning. |
| Agreement task | Precomputed relationship plus actual response/correctness | A wrong answer remains a real behavioral outcome; do not erase it. |
| Effort/preference/activation | Actual post-trial self-report | Keep the separate rating scales; missing is not neutral. |
| Intention | Actual attempted intention and completion report | Preserve requested cue and disagreements; uncertain or incomplete attempts are not clean labels. |
| Artifact | Observation, calibration task, signal review | A requested movement time is not a verified movement time. |
| Free choice | Private choice reported after a timestamped prediction | Do not reveal the choice to the model before prediction. |

Define exclusion rules before fitting. Log both all attempted trials and the clean subset, with reasons and counts per class/day. Report performance with coverage so that rejecting most difficult trials cannot masquerade as a reliable interface.

## 9. Decoder development design

This is an implementation plan for the receiving agent, not supplied trained software.

### 9.1 Signal preparation

Preserve raw data. Verify actual sample rate, units, reference, missing data, and clipping before making a filtered copy. Do not treat Bluetooth arrival intervals as EEG sampling intervals. Inspect T01-T08 and representative cognitive trials.

For an initial sustained-activity analysis, consider a conservative band such as 1-30 Hz **only when supported by the verified sample rate, device response, and target**. Document the filter and avoid filtering across packet gaps or trial boundaries. Choose causal filtering for a real-time model and evaluate it as such. Do not validate an offline zero-phase filter and then assume identical online behavior. Do not add a 50/60 Hz notch by default when the analysis passband already removes that frequency; use measured noise and the verified Nyquist limit to make that choice.

With one independent channel, start with robust per-window amplitude summaries, log power in supported frequency bands, relative power with documented denominators, and a small set of temporal features. Candidate bands include theta 4-7 Hz, alpha 8-12 Hz, and beta 13-25 Hz; these are descriptive ranges, not universal mental-state labels. Omit unsupported or contaminated bands. Avoid interpreting high-frequency jaw/muscle activity as cognition.

Use the calibration examples to set artifact rules in known units or validated device units. Check flat lines, rails/clipping, large transients, dropouts, and eye/movement patterns. Single-channel data cannot reliably separate every artifact from neural activity. If contamination predicts the class, report the confound. Keep baselines and artifact windows out of the cognitive feature set unless a predeclared normalization uses the baseline.

### 9.2 Fit the smallest useful model

Train separate models for distinct questions. Do not train one classifier mixing `fire`, `hard`, `yes`, and `pleasant` as if these were mutually exclusive states.

- **Primary command model:** YES / NO / no command. MORE and PAUSE remain exploratory until independently supported. Include T55 negatives; retain T16 as a command-word-reading control and T56 as a distinct free-choice assay.
- **Task-demand model:** compare matched task families and the reported effort scale. A model trained only on arithmetic should not be described as a universal attention meter.
- **Preference model:** predict recorded ratings or a preregistered coarse grouping. If ratings do not vary, there is no useful target to learn.
- **Stimulus concept model:** first test water/fire/dog/piano within modality. Report which classes and modalities were actually evaluated. Then attempt cross-modal transfer with adequate balanced data; do not silently combine incomparable accuracies.

Start with regularized logistic regression or another simple, inspectable classifier. Fit scaling, feature selection, normalization parameters, probability calibration, and hyperparameters using training groups only. Choose abstention thresholds on development validation, then freeze them. A predeclared, label-free per-session baseline adjustment may use that session's calibration interval; keep the adjustment rule fixed and log its inputs. Do not fit that adjustment to held-out task labels.

For EEG-only experiments, model inputs must not contain trial ID, test ID, stimulus identity, filename, cue word, answer, rating, response latency, requested class, trial order, or future samples. If the agent uses task context, evaluate and name that as a separate context-assisted model, and compare with a context-only baseline.

If dividing a trial into multiple windows, keep every window from the same trial in one split. Group by session/day and, for new-stimulus claims, source stimulus family. Do not allow overlapping samples, alternate crops of the same sound, duplicate images, or training-derived transforms of test data to cross a claimed independence boundary. [R8]

### 9.3 Evaluation required before a translation claim

Report counts, class distributions, balanced accuracy, a confusion matrix, macro-F1 when appropriate, abstention coverage, and per-day results. For rating targets report an appropriate error/correlation and the observed rating range. Use uncertainty estimates that respect day/trial grouping; thousands of overlapping windows are not thousands of independent observations.

Compare with a majority-class predictor, chance/permutation analysis appropriate to the design, and context-only baselines where relevant. For a balanced four-class concept task, nominal chance accuracy is 25%; this does not establish significance or practical usefulness. For commands, distinguish errors among classes from false commands emitted during rest.

Evaluate all of the following separately:

1. Familiar stimuli on a new day: add frozen-model replays from development cards in locked test sessions if this claim is needed.
2. New stimuli on a new day: T57-T62. These cover selected class/modality combinations, not every combination.
3. New command wording: T63-T64. These still have cues.
4. Uncued private choice: T56 with later-session EEG withheld from fitting if it is used for a generalization claim.
5. No-command operation: the later-session T55 trials plus a separately marked 5-minute quiet no-command recording. Keep normal blinks and ordinary small movement in this idle evaluation so false activation is measured honestly. This is an additional recording, not a 65th test type.

Before invoking the agent from a continuous decoder, measure idle false activations per minute and missed-command rates using the exact deployment smoothing/refractory rules. Include more realistic no-command contexts in subsequent evaluation. A classifier that distinguishes externally shown images may have learned sensory properties; a cued-command classifier may have learned cue residue or a mental strategy. Neither establishes access to arbitrary thoughts.

Do not run a language model to turn a low-confidence category estimate into a detailed sentence in the participant's voice. Repeated failures to exceed baseline are a valid study outcome. Narrow the target or improve acquisition rather than extending the vocabulary on unsupported evidence.

## 10. Agent integration contract

The interface should expose a cautious **candidate signal**, with the participant able to correct it. Unknown data, poor quality, missing timing, an unsupported target, or failure to meet the validated decision rule yields `uncertain`. Do not force a class.

Example output shape below contains placeholders, not actual measurements or calibrated probabilities:

```json
{
  "schema_version": "1.0",
  "participant_id": "local_participant_01",
  "model_version": null,
  "scope": "intentional_yes_no_rest",
  "window_start_acquisition_s": null,
  "window_end_acquisition_s": null,
  "signal_quality": "unverified",
  "candidate_label": "uncertain",
  "class_probabilities": null,
  "probabilities_calibrated": false,
  "decision_rule_version": null,
  "recording_context_supported": false,
  "participant_confirmation": null
}
```

If a validated model later emits `yes`, present it as a possible intended YES with its limitations. It is not consent to an unrelated action. The receiving agent must use the normal authorization rules for consequential actions; an EEG estimate must not grant new permission. A candidate PAUSE may pause stimulus presentation as a reversible comfort measure, while a physical/manual stop remains available throughout.

Log the prediction before collecting the participant's correction. Store that correction separately. Do not immediately retrain a production model on each correction: first review labels and quality, create a versioned development dataset, retrain offline, and rerun independent evaluation. Do not let the agent's guessed interpretation overwrite ground truth.

For state support, prefer offers such as "Would you like a break?" to declarations such as "You are stressed." Keep self-reports authoritative about subjective experience. This program evaluates an interface; it does not diagnose health conditions or establish a general ability to read the participant's mind.

## 11. Completion criteria for the receiving agent

For the current request, this document is the deliverable. Future execution requires the existing reader and participant-controlled recordings. When implementing it, produce:

- A verified hardware/export note with unresolved fields explicitly marked.
- An approved, cached stimulus ledger with licenses, attributions, exact excerpts, and hashes.
- A recorded session/trial/marker dataset with preserved raw EEG and transparent exclusions.
- A split manifest fixed before training, separating development and locked evaluation.
- A short result report stating which narrow targets worked, failed, or remain untested, with counts and baselines.
- An agent output contract that preserves uncertainty and participant correction.

If hardware data, prepared media, timing, or independent validation is missing, report that concrete gap. Do not claim to have completed collection, trained a model, or established brain translation from this specification alone.

## 12. Exact candidate media ledger

Records below were returned by the Openverse API on 2026-10-13. License statements are indexed metadata, pending original-source verification. Image compositions were inspected; audio was not auditioned. No file hashes or approved excerpts have been generated. The URLs are retrieval candidates, not a guarantee of future availability.

### 12.1 Openverse image records

#### `IMG_WATER_A` - Flowing Stream.

- **Role:** development; **status:** candidate.
- **Creator:** Jake Cook; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `1d0bc66e-731d-46bf-b007-d84211fa0275`; [API record](https://api.openverse.org/v1/images/1d0bc66e-731d-46bf-b007-d84211fa0275/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/75480089@N04/8470713430).
- **Media URL:** [image file](https://live.staticflickr.com/8518/8470713430_fcaf126785_b.jpg).
- **Retain attribution:** "Flowing Stream." by Jake Cook is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_FIRE_A` - Fireplace fire

- **Role:** development; **status:** candidate.
- **Creator:** johanleijon; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `46d4ccdb-9356-488b-9072-0d356c9c65a6`; [API record](https://api.openverse.org/v1/images/46d4ccdb-9356-488b-9072-0d356c9c65a6/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/30981697@N03/3345153572).
- **Media URL:** [image file](https://live.staticflickr.com/3415/3345153572_f85ebd9204_b.jpg).
- **Retain attribution:** "Fireplace fire" by johanleijon is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_DOG_A` - Soppy Dog

- **Role:** development; **status:** candidate.
- **Creator:** JaseCurtis; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `9dde58f1-d1f4-4f41-8ab9-c35f8f6b7d17`; [API record](https://api.openverse.org/v1/images/9dde58f1-d1f4-4f41-8ab9-c35f8f6b7d17/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/25722571@N08/6090548367).
- **Media URL:** [image file](https://live.staticflickr.com/6191/6090548367_5491b87c28_b.jpg).
- **Retain attribution:** "Soppy Dog" by JaseCurtis is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_PIANO_A` - Piano

- **Role:** development; **status:** candidate.
- **Creator:** Sean MacEntee; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `c9506407-6291-47b6-9faf-30906bd41a87`; [API record](https://api.openverse.org/v1/images/c9506407-6291-47b6-9faf-30906bd41a87/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/18090920@N07/5918579324).
- **Media URL:** [image file](https://live.staticflickr.com/6012/5918579324_c9262fa879_b.jpg).
- **Retain attribution:** "Piano" by Sean MacEntee is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_FOREST_A` - Forest Path

- **Role:** development; **status:** candidate.
- **Creator:** Yixler; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `1779e8f4-2c8a-4bb1-a861-2805fa3c93d4`; [API record](https://api.openverse.org/v1/images/1779e8f4-2c8a-4bb1-a861-2805fa3c93d4/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/40978147@N02/3772968946).
- **Media URL:** [image file](https://live.staticflickr.com/2583/3772968946_3fd9515ee4_b.jpg).
- **Retain attribution:** "Forest Path" by Yixler is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_STREET_A` - New York City street

- **Role:** development; **status:** candidate.
- **Creator:** Zeeyolq Photography; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `ff9c6803-e6a0-4fb8-9504-3cf83c07645a`; [API record](https://api.openverse.org/v1/images/ff9c6803-e6a0-4fb8-9504-3cf83c07645a/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/73422502@N08/11115347443).
- **Media URL:** [image file](https://live.staticflickr.com/5477/11115347443_c746b25f0a_b.jpg).
- **Retain attribution:** "New York City street" by Zeeyolq Photography is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_FLOWERS_A` - Several white tulips in a flower garden

- **Role:** development; **status:** candidate.
- **Creator:** Horia Varlan; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `ce36b34b-2640-4137-8978-68d083c87261`; [API record](https://api.openverse.org/v1/images/ce36b34b-2640-4137-8978-68d083c87261/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/10361931@N06/4290838604).
- **Media URL:** [image file](https://live.staticflickr.com/4044/4290838604_5338415100_b.jpg).
- **Retain attribution:** "Several white tulips in a flower garden" by Horia Varlan is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_TOOLS_A` - Hard Hat with Hand Tools Inside

- **Role:** development; **status:** candidate.
- **Creator:** toolstop; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `5fb2060f-82ce-4470-84e0-8a8657c1987c`; [API record](https://api.openverse.org/v1/images/5fb2060f-82ce-4470-84e0-8a8657c1987c/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/42408834@N06/4420180838).
- **Media URL:** [image file](https://live.staticflickr.com/4048/4420180838_971954ef00_b.jpg).
- **Retain attribution:** "Hard Hat with Hand Tools Inside" by toolstop is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_DOG_H` - Dog

- **Role:** reserved evaluation; **status:** candidate.
- **Creator:** Sean MacEntee; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `19e2e417-a819-4761-b838-d648b92d48ce`; [API record](https://api.openverse.org/v1/images/19e2e417-a819-4761-b838-d648b92d48ce/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/18090920@N07/6114260088).
- **Media URL:** [image file](https://live.staticflickr.com/6198/6114260088_20a2cc5ae5_b.jpg).
- **Retain attribution:** "Dog" by Sean MacEntee is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

#### `IMG_PIANO_H` - Piano

- **Role:** reserved evaluation; **status:** candidate.
- **Creator:** Molly DG; **indexed license:** BY 2.0 ([terms](https://creativecommons.org/licenses/by/2.0/)).
- **Openverse ID:** `d33bf083-24c9-44db-ac76-07f8fd038c05`; [API record](https://api.openverse.org/v1/images/d33bf083-24c9-44db-ac76-07f8fd038c05/).
- **Original source:** [flickr landing page](https://www.flickr.com/photos/54400117@N03/5068517177).
- **Media URL:** [image file](https://live.staticflickr.com/4146/5068517177_63f0406105_b.jpg).
- **Retain attribution:** "Piano" by Molly DG is licensed under CC BY 2.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/2.0/.

### 12.2 Freesound recordings discovered through Openverse

#### `AUD_WATER_A` - Water stream 5

- **Role:** development; **status:** candidate.
- **Creator:** Kolezan; **indexed license:** BY 4.0 ([terms](https://creativecommons.org/licenses/by/4.0/)).
- **Openverse ID:** `1815cfdb-6789-47f2-a818-a975ddc246f3`; [API record](https://api.openverse.org/v1/audio/1815cfdb-6789-47f2-a818-a975ddc246f3/).
- **Original source:** [freesound landing page](https://freesound.org/people/Kolezan/sounds/132049).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/132/132049_29193-hq.mp3).
- **Indexed source duration:** 60.000 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Water stream 5" by Kolezan is licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/.

#### `AUD_FIRE_A` - Fire Crackling 01.wav

- **Role:** development; **status:** candidate.
- **Creator:** kingsrow; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `166449d3-2ad9-4f48-bbc1-e15887464a56`; [API record](https://api.openverse.org/v1/audio/166449d3-2ad9-4f48-bbc1-e15887464a56/).
- **Original source:** [freesound landing page](https://freesound.org/people/kingsrow/sounds/181563).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/181/181563_1857065-hq.mp3).
- **Indexed source duration:** 34.210 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Fire Crackling 01.wav" by kingsrow is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_DOG_A` - A dog barking (close recording)

- **Role:** development; **status:** candidate.
- **Creator:** felix.blume; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `f3344937-8ed4-4bfc-ad1a-1499ea986f02`; [API record](https://api.openverse.org/v1/audio/f3344937-8ed4-4bfc-ad1a-1499ea986f02/).
- **Original source:** [freesound landing page](https://freesound.org/people/felix.blume/sounds/199261).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/199/199261_1661766-hq.mp3).
- **Indexed source duration:** 20.705 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "A dog barking (close recording)" by felix.blume is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_PIANO_A` - Summer night piano solo

- **Role:** development; **status:** candidate.
- **Creator:** szegvari; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `f65774bd-72f9-4344-a33b-597e98e6db1a`; [API record](https://api.openverse.org/v1/audio/f65774bd-72f9-4344-a33b-597e98e6db1a/).
- **Original source:** [freesound landing page](https://freesound.org/people/szegvari/sounds/577844).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/577/577844_2282212-hq.mp3).
- **Indexed source duration:** 52.000 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Summer night piano solo" by szegvari is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_BIRDS_A` - Woods ambience in Missouri, some birds singing and soft breeze.

- **Role:** development; **status:** candidate.
- **Creator:** felix.blume; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `eddd4878-7d54-4e61-a0e8-7d776b7e1d36`; [API record](https://api.openverse.org/v1/audio/eddd4878-7d54-4e61-a0e8-7d776b7e1d36/).
- **Original source:** [freesound landing page](https://freesound.org/people/felix.blume/sounds/665156).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/665/665156_1661766-hq.mp3).
- **Indexed source duration:** 576.842 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Woods ambience in Missouri, some birds singing and soft breeze." by felix.blume is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_TRAFFIC_A` - Hamburg City Traffic Ambience

- **Role:** development; **status:** candidate.
- **Creator:** Kinoton; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `a9339c75-5813-4277-9aee-5d589ddc36f3`; [API record](https://api.openverse.org/v1/audio/a9339c75-5813-4277-9aee-5d589ddc36f3/).
- **Original source:** [freesound landing page](https://freesound.org/people/Kinoton/sounds/352152).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/352/352152_2247456-hq.mp3).
- **Indexed source duration:** 234.322 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Hamburg City Traffic Ambience" by Kinoton is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_RAIN_A` - Gentle Rain Mix (2018)

- **Role:** development; **status:** candidate.
- **Creator:** esh9419; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `fa70e716-4cca-4b49-b0a7-3589978035ea`; [API record](https://api.openverse.org/v1/audio/fa70e716-4cca-4b49-b0a7-3589978035ea/).
- **Original source:** [freesound landing page](https://freesound.org/people/esh9419/sounds/417616).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/417/417616_3285887-hq.mp3).
- **Indexed source duration:** 720.000 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Gentle Rain Mix (2018)" by esh9419 is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_MACHINE_A` - machine hum

- **Role:** development; **status:** candidate.
- **Creator:** AlaskaRobotics; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `09d5e4a3-8398-43e5-b453-c2a2f8271b3f`; [API record](https://api.openverse.org/v1/audio/09d5e4a3-8398-43e5-b453-c2a2f8271b3f/).
- **Original source:** [freesound landing page](https://freesound.org/people/AlaskaRobotics/sounds/221566).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/221/221566_1282865-hq.mp3).
- **Indexed source duration:** 14.832 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "machine hum" by AlaskaRobotics is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

#### `AUD_WATER_H` - Water Stream Dripping.wav

- **Role:** reserved evaluation; **status:** candidate.
- **Creator:** digifishmusic; **indexed license:** BY 4.0 ([terms](https://creativecommons.org/licenses/by/4.0/)).
- **Openverse ID:** `e7c4fb41-75fa-4408-89b2-dc8d7b5f76cd`; [API record](https://api.openverse.org/v1/audio/e7c4fb41-75fa-4408-89b2-dc8d7b5f76cd/).
- **Original source:** [freesound landing page](https://freesound.org/people/digifishmusic/sounds/33772).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/33/33772_29541-hq.mp3).
- **Indexed source duration:** 28.485 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "Water Stream Dripping.wav" by digifishmusic is licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/.

#### `AUD_FIRE_H` - fire crackles

- **Role:** reserved evaluation; **status:** candidate.
- **Creator:** florianreichelt; **indexed license:** CC0 1.0 ([terms](https://creativecommons.org/publicdomain/zero/1.0/)).
- **Openverse ID:** `914bb44e-4396-42e3-b369-b8d5ac5b793c`; [API record](https://api.openverse.org/v1/audio/914bb44e-4396-42e3-b369-b8d5ac5b793c/).
- **Original source:** [freesound landing page](https://freesound.org/people/florianreichelt/sounds/563766).
- **Media URL:** [Freesound MP3 preview](https://cdn.freesound.org/previews/563/563766_6253486-hq.mp3).
- **Indexed source duration:** 50.917 s; **trial excerpt:** 8 s, start/end not yet selected; **uploader AI preference:** not available in retrieved index.
- **Retain attribution:** "fire crackles" by florianreichelt is marked with CC0 1.0. To view the terms, visit https://creativecommons.org/publicdomain/zero/1.0/.

## 13. Sources and limits

These references support the design constraints and media workflow, not a claim that this specific headband has been validated for them. Research summaries are deliberately limited; the rest of the protocol is an original proposed study design.

- **[R1]** [Towards decoding individual words from non-invasive brain recordings, Nature Communications (2025)](https://www.nature.com/articles/s41467-025-65499-0). Research demonstrates experimental word-decoding approaches across EEG/MEG datasets, with performance depending on device, task, and data. It does not establish unrestricted thought reading from a consumer headband.
- **[R2]** [Nordic Semiconductor: Buttonless Secure DFU Service](https://docs.nordicsemi.com/r/bundle/nrf5_sdk_v17.1.0/page/service_dfu.html). Identifies `0xFE59` as a firmware-update service; it does not specify this headband's EEG data protocol.
- **[R3]** [Lab Streaming Layer: Time Synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html). Describes clock domains, offsets, synchronization metadata, and XDF-based recording/import considerations.
- **[R4]** [Openverse API documentation](https://api.openverse.org/v1/) and [Openverse API consumer documentation](https://docs.openverse.org/api/). Live image and Freesound-source audio queries were used to retrieve the candidate records above. [Openverse's license-verification caution](https://docs.openverse.org/_preview/3965/api/reference/made_with_ov.html) explains why original-source checks remain necessary.
- **[R5]** [Freesound API authentication](https://freesound.org/docs/api/authentication.html). Documents token authentication and OAuth2.
- **[R6]** [Freesound API resource reference](https://freesound.org/docs/api/resources_apiv2.html). Documents search, previews, original-file downloads, metadata fields, and `gen_ai_preference`.
- **[R7]** [Freesound FAQ: generative AI preferences](https://freesound.org/help/faq/). Explains uploader preferences in addition to sound licenses.
- **[R8]** [Data leakage in deep learning studies of translational EEG (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/). Shows how dependence between EEG segments can inflate evaluation. This protocol applies the same independence concern to sessions, repeated stimuli, and overlapping within-person windows.

**Scope reminder:** 64 test types, ten image candidates, ten Freesound audio candidates, original reading passages, label/timing contracts, and a decoder-development plan. Exact device signal specifications, synchronized presentation, prepared audio excerpts, collected EEG, model fitting, and independent validation remain future work.
