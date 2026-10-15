# Hardware, electrode prep and session procedure

Applies to the **NingNao / SereniBrain TH21A** headband. The reverse-engineering
notes (GATT services, handshake frames, packet layout, the Just-Works pairing)
live in `../../ble/README.md`; this document is the *operator* procedure.

## The device

| | |
|---|---|
| Advertised name | `TH21A_D937B53E3FCC` |
| BLE address | per-unit *static random*, e.g. `AA:BB:CC:DD:EE:FF` — find yours with `nb doctor --scan`, or leave it unset and the band is discovered by its `TH21A` name |
| Channel | **one**, dry frontal, referenced on the band |
| Rate | 250 Hz nominal, 10 samples per notification, ~25 notifications/s |
| Resolution | 24-bit signed ADC counts, **uncalibrated** (no microvolt scale) |
| Rails | saturates at ±8388607 — this is what "electrodes off the head" looks like |
| Contact check | `SELF_CHECK` reply contains four `OK` tokens when all electrodes are seated |
| Pairing | no passkey (legacy Just Works); the link key lives in `/var/lib/bluetooth/...` |

Because the amplifier is uncalibrated, **every number NeuroBridge reports is
relative** (percentage of 1–45 Hz power, or a log-ratio, or a z-score against
your own baseline). That is not a limitation we work around; it is the reason
the pipeline is built the way it is.

## Before every session

1. **Clean the electrodes.** Dry electrodes accumulate sebum and stop
   conducting. Wipe each electrode with an alcohol pad or a little saline on a
   tissue, then let it dry for a few seconds. This single step fixes the
   majority of bad recordings.
2. **Clean your skin.** Wipe the forehead and the nose bridge with an alcohol
   pad. Oily skin at Fpz is the main source of slow drift.
3. **Seat the band.** Switch button on the bridge of the nose; front edge just
   above the eyebrows; all electrodes flat on skin. **No hair trapped under any
   electrode** — that is electrical noise, not brain.
4. **Power on and wait.** Long-press the power button ~3 s. The headband
   advertises for a while; if the session fails to find it, power-cycle and
   retry.
5. **Confirm contact** before recording. Either run `nb doctor --scan`, or just
   start a session — the runner logs the `SELF_CHECK` string, and the quality
   report at the end of the session tells you whether the capture was usable.

## During a session

* Sit still, jaw relaxed, minimise swallowing for the duration of a block.
* Blink normally, except in the blink-control tests where blinks are the signal.
  Frontally, a blink is a large slow deflection; if you suppress blinking the
  eyes dry out and you generate slow drift instead — worse than the blinks.
* Do not talk during tests that are not about speech. Jaw and neck muscle
  activity is broadband EMG and it dominates this electrode site.
* Glance at the screen only when the protocol asks for it. Watching a progress
  bar produces eye movement, and eye movement is a big frontal artefact.
* Take the scheduled breaks. Dry-electrode drift grows with time-on-task, and
  the tier-1 tests are specifically measuring that drift as *fatigue*.

## Screen hygiene (avoiding visual distraction)

The stimulus window is fullscreen by default and the dark NeuroBridge surface
stays up for the whole session — instructions, cues and breaks included — so the
desktop is never visible between phases. The cursor is hidden and the
screensaver/display blanking is inhibited, then released when the session ends.

Belt and braces, because a changing desktop is exactly the kind of thing that
costs you a block:

1. **Turn off a rotating wallpaper.** MATE: *Menu ▸ Control Center ▸ Appearance ▸
   Background*, and set the change interval to *Never* (or the same image). A
   wallpaper that changes mid-block is a visual event your visual cortex will
   happily process for you.
2. **Silence notifications.** *Control Center ▸ Notifications* → turn off
   banners, or enable *Do Not Disturb*. A popup over a stimulus is worse than a
   missing trial.
3. **Close other windows** and, if you use a screensaver with a lock, make sure
   it is not set to trigger inside a block; the runner inhibits it, but the lock
   screen may still win on a policy setting.
4. **Set the display to its native resolution** and leave
   `nb run`'s default fullscreen on (`--windowed` exists for debugging only).
5. The runner records which screensaver-inhibition method worked in the session
   metadata (`screensaver_inhibit`); if it says `none`, set the screensaver to a
   long timeout manually before a long session.

## Microphone setup (voice channel)

Needed for `--mic` sessions: reading aloud, spoken responses, fluency and
onset-latency measurement. Reading is mostly behaviour and a frontal dry
electrode cannot see behaviour, which is exactly why this channel exists.

1. **Check the input device before the session**: `./bin/nb doctor --mic`. It
   lists input devices and records 2 s (announced in its output) to answer the
   only question that matters — is the microphone actually picking anything up?
   The sample is discarded unless you ask to keep it.
2. **Position**: a headset or lapel microphone is far better than a laptop
   array, because the array also picks up the speakers and the room. Keep the
   microphone off the headband strap so cable movement does not couple into the
   electrodes.
3. **Level**: aim for speech peaks around −12 to −6 dBFS and a room floor below
   −45 dBFS. `nb voice --record 20` writes a sample you can inspect; the
   session summary prints the achieved RMS.
4. **Fix the output volume before a loudness or audio test** and note it.
   `nb doctor` reports the output latency; `voice.measure_roundtrip()` measures
   the full output+acoustic+input delay, which is what makes an onset latency
   interpretable.
5. **Privacy**: capture is opt-in per session, the audio stays in
   `data/sessions/<id>/audio.wav` on this machine, and consent with a timestamp
   is recorded in the session metadata. Nothing is uploaded. If a session
   should not keep audio, run it without `--mic`.

The microphone is also the honest check on a known confound: reading aloud
moves the jaw, and jaw EMG is large at Fpz. If the audio envelope and the EEG
gamma band track each other trial by trial, the classifier learned the jaw
rather than the cortex — and now you can see that instead of assuming it away.

## Session checklist (`nb run` does most of this for you)

```
[ ] electrodes wiped, skin wiped
[ ] band seated: nose-bridge switch, above eyebrows, no hair under electrodes
[ ] headband powered, LED confirmed, within ~1 m of the Bluetooth adapter
[ ] other Bluetooth/Wi-Fi-heavy apps closed (marker jitter)
[ ] system volume fixed and noted (matters for the loudness tests)
[ ] if reading aloud or speaking: nb doctor --mic passes, mic positioned off the band
[ ] session budget chosen: 45 min, one to four tests
[ ] nb doctor passes the display + audio checks
[ ] nb run <ids> -y            (add --mic when a voice channel is wanted)
[ ] after the run: read the quality verdict and the jitter report
[ ] if --mic: nb voice --session latest
```

## Reading the quality verdict

`nb` prints a verdict per session and stores it in `summary.json`:

| Verdict | Meaning | What to do |
|---|---|---|
| `GOOD` / `FAIR` | usable; artefact levels are sane | analyse as normal |
| `MARGINAL` | artefact heavy | treat results as exploratory; reseat before the next session |
| `POOR` | saturated or drift-dominated | **discard**; clean the electrodes, reseat, re-run |

`POOR` captures are excluded from training by the pipeline. This is deliberate:
a decoder trained on saturated captures learns the saturation artefact, which
looks great in cross-validation and is worthless in the real world.

## The alpha reactivity check (your instrument's sanity test)

Eyes-closed alpha (~8–13 Hz, maximal over the back of the head but visible at
Fpz with a single dry electrode) rising on eye closure is the cheapest
available proof that the channel carries cortical signal rather than electrode
artefact. Your own earlier measurement on this headband:

| Phase | alpha share |
|---|---|
| eyes open | 4.5 % |
| eyes closed | **8.9 %** (×1.92) |
| eyes open | 4.8 % |

If `T003` does not reproduce something like that, stop and fix the contact
before running anything else. A session where alpha does not move is a session
where nothing else can be trusted.

## Troubleshooting, in order of likelihood

| Symptom | Cause | Fix |
|---|---|---|
| Not advertising | asleep / drained | long-press 3 s; charge |
| `SELF_CHECK` shows fewer than four `OK` | poor contact | clean + reseat; check for hair |
| Delta > 90 %, saturation flagged | off the head, or completely dry electrodes | clean with saline, reseat |
| Large slow swings every few seconds | electrode movement / jaw | reseat the band tighter, relax the jaw |
| 50/60 Hz spike in the spectrum | mains pickup, poor grounding | move away from chargers and PSUs; the pipeline notches it but it should not be there |
| Marker jitter > 50 ms | Bluetooth congestion, CPU load | close other apps, disable Wi-Fi power saving |
| Effective fs drifts from 250 Hz | dropped BLE packets | normal at low rates; the session stats report gaps — if `missing_samples` is large, move closer to the adapter |
| Audio feels late relative to the visual cue | output latency | run `T004`, then set `output_latency_offset_s` in `neurobridge.yaml` |
| `--mic` session has no `audio.wav` | microphone busy, muted, or consent refused | `nb doctor --mic`; check nothing else holds the device (browsers do) |
| Speech onset latencies look absurd (negative or >1 s) | wrong device, or the mic clock drifted | re-run `voice.measure_roundtrip()`; keep the same input device across a session |
| Audio clips (peak at 1.0) | input gain too high | lower the input level; check `clipped_pct` in the session summary |

## Safety and scope

This is a consumer biofeedback device and a research-grade recording setup.
NeuroBridge is **not a medical device**, and nothing it reports should be used
for diagnosis, treatment, or any decision that matters. Stop a session if you
feel discomfort, dizziness, or skin irritation, and if a session makes you
anxious, take a break — no result here is worth that.
