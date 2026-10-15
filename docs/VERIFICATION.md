# Verification record

What was actually tested, with the commands and the numbers. Gaps are listed in
§8 rather than glossed over.

Everything below runs on this machine with no headband attached, using the
project venv (`.venv/bin/python`).

---

## 1. Acquisition is compatible with the reader you already built

The headband cannot be used as a reference implementation, but your **existing
captures** can. `../ble/session_packets.txt` is a real 60 s recording
(eyes-open / eyes-closed / eyes-open) made with the original `ble/` tools.

```
$ .venv/bin/python -c "... compare neurobridge.device.th21a.parse_packet with the ble/ decoder ..."
neurobridge samples: 15060 | original decoder samples: 15060
identical values: True
index counter monotonic-by-1 fraction: 1.000
index wraps: 60
amplitude range: 659864 .. 961335   ptp 301471
```

**Every one of 15,060 decoded samples is byte-identical to the original
decoder**, and the 8-bit sample index increments by exactly 1 between
consecutive samples across 60 wraps — i.e. the packet layout in
`device/th21a.py` is right, not approximately right.

Importing that capture as a NeuroBridge session:

```
$ .venv/bin/python scripts/import_ble_packets.py ../ble/session_packets.txt --test-id T003
session_packets.txt: 1506 packet lines, 3 contiguous phase run(s), 15060 samples (60.2s at 250 Hz)
runs per phase: eyes_open x2, eyes_closed x1
alpha: open 2.76%  closed 6.18%  ratio 2.24   -> STRONG - this is real EEG
SIGNAL QUALITY: GOOD (4/4, dry_frontal thresholds)
```

and evaluating it end to end through the pipeline:

```
$ ./bin/nb eval T003 --session import-session_packets-T003
alpha_ratio_closed_open  1.911  gate 1.3  -> PASS
```

`1.911` is an independent reproduction of the **×1.92** you measured in the
original session — two different implementations, two different windowings,
same answer. That is the single most reassuring number in this document: the
analysis chain sees your cortex, not your electrode.

## 2. Quality thresholds were recalibrated against real data

The first version of `device/quality.py` inherited a wet-multi-channel intuition
(drift < 3, delta share < 60 %) and graded that same verified capture
**MARGINAL 1/3** — which would have blocked training on good data. Measured on
your recording: saturation 0.000 %, drift ratio 7.07, delta share 81.2 %, motion
0.00 %, yet alpha ×2.24 on eye closure.

Dry frontal electrodes genuinely carry most of their energy below 4 Hz, so
`dry_frontal` (the default, with `wet_multichannel` kept for other montages) now
uses drift < 15, delta < 90 %, motion < 3 %, saturation < 0.5 % with a hard fail
above 5 %, and treats alpha reactivity ≥ 1.5 as positive evidence.

```
SIGNAL QUALITY: GOOD (4/4, dry_frontal thresholds)
  - saturation 0.000% (ok, <= 0.5%)
  - drift ratio 7.07 (ok, < 15.0)
  - delta share 81.2% (ok, < 90.0%)
  - motion 0.00% of samples jump >20x median (ok, < 3.0%)
  - alpha x2.24 on eye closure - real cortical signal is visible on this channel
```

Same data, honest verdict, and the saturation hard-fail still catches an
electrode-off-the-head capture.

## 3. Catalog

```
$ ./bin/nb catalog validate
OK all 64 specs valid
$ ./bin/nb catalog audit
no structural problems found
```

`audit` checks what the schema cannot: every spec's design actually builds a
timeline (30 trial-shaped, 34 condition-shaped), every feature string is in the
implemented vocabulary, every gate sits inside the band implied by the spec's
`feasibility`, no single-class test carries an accuracy gate, and it reports the
declared-vs-design-derived duration disagreements (planning books the larger:
**768 min ≈ 17 sittings at 45 min**).

Stimuli resolve offline for every test:

```
$ .venv/bin/python -c "<prepare() every spec with allow_download=False>"
total items resolved offline: 165
failures: none
```

Real network paths were also exercised: Openverse search + download returned
PIL-validated, downscaled, licence-attributed images with zero warnings; the
Freesound client returns `NoTokenError` immediately without a token and the
builder substitutes synthesised audio with an explicit warning naming the
generator.

## 4. End-to-end on synthetic EEG (the leakage-checked part)

`nb simulate` builds a session through the same store format the headband
writes, then runs the ordinary pipeline over it. Because the synthetic source
injects a *known* response, this answers "if the brain produced the effect,
would this code find it?".

**Decoding results** (per-test duration scale; each eval includes the spec's
pre-registered permutation test unless noted):

| test | injected | epochs | bal-acc | chance | AUC | verdict |
|---|---|---|---|---|---|---|
| T301 auditory oddball | P300 | 240 | **1.000** | 0.500 | 1.000 | decoded, gate PASS |
| T003 alpha reactivity | alpha bands | 120 | **1.000** | 0.500 | 1.000 | decoded, gate PASS |
| T007 blink/clench/rest | artefacts | 75 | **1.000** | 0.333 | – | decoded |
| T101 sustained attention | P300 on targets | 380 | **0.873** | 0.500 | 0.974 | decoded |
| T201 n-back load | theta/alpha/beta | 7 | **0.833** | 0.500 | 0.917 | decoded, gate PASS |
| T103 flicker entrainment | none | 26 | 0.452 | 0.333 | – | null-ok |
| T008 sham control | none | 117 | **0.437** | 0.500 | 0.319 | null-ok (below chance) |
| T002/T004/T005 | – | – | n/a | – | – | single-class by design, quality/SNR gates |

**The leakage check that matters.** Same test, same duration scale, same
`n=240`, same four blocks — the only difference is whether the synthetic source
injected a response:

| T301, scale 0.2, seed 99, n=240 | bal-acc | chance | AUC | verdict |
|---|---|---|---|---|
| effect injected | **0.908** | 0.500 | 0.972 | PASS (gates 0.70 / 0.75) |
| **nothing injected** | **0.500** | 0.500 | 0.506 | FAIL (gates missed) |

```
$ ./bin/nb simulate T301 --scale 0.2               # effect present
OK T301: bal-acc 0.908 (chance 0.5), auc 0.972, n=240 -> PASS   [28 s, idle machine]

$ ./bin/nb simulate T301 --scale 0.2 --effect-size 0 --seed 99   # null
FAIL T301: bal-acc 0.500 (chance 0.5), auc 0.506, n=240 -> FAIL
```

Exactly 0.500 when there is nothing to find, 0.908 when there is. There is no
leakage in the chain — and the `FAIL` in the second row is the *desired*
outcome, which is why control tests are read as inverted.

Note also that 0.908 is a more credible number than the 1.000 seen at some
scales: a decoder that scores perfectly on every run is usually a decoder that
has found the generator, not the signal.

**Three bugs this found and forced fixes for:**

1. T201–T207 declare a 60 s condition run *and* a 1.2 s stimulus SOA. Treating
   them as short trials made every 4 s analysis window straddle both conditions
   and averaged the injected effect away (T201 sat at 0.513 ≈ chance). Fixed by
   letting a declared condition-run duration outrank a declared SOA:
   `simulate.has_condition_runs`.
2. A global duration scale starves windowed tests: at scale 0.05, T003's 20 s
   phases became 1 s and produced 4 windows total. Fixed with a per-test scale
   that compresses only as far as `min_cond_s` allows.
3. In the analysis itself: repeated **same-label** class markers (the refresh
   markers that keep a long condition labelled) were treated as class *changes*,
   so most windows inside a long condition were discarded — T201's windowed
   epoching yielded 8 windows instead of 16. Same-label repeats now extend the
   condition rather than ending it.

Two further bugs were caught in the voice module by their own tests: a
float32 RMS accumulator that overflowed to `Inf` on an input glitch, and
`Session.latest()` returning a stale synthetic session because `"sim-"` sorts
after `"s-"` lexicographically. Both are fixed (`float64` accumulation;
ordering by session start time).

## 5. Presentation runner

Independently verified by the runner workstream across all design shapes, with
the real timing paths exercised:

| check | result |
|---|---|
| T301 scale 0.02, limit 24 | 1750 samples, 6.96 s, **251.3 Hz effective**, 91 events, 24/24 trials, quality GOOD (4/4) |
| marker jitter | mean interval 77.4 ms, **sd 7.2 ms, max 20.5 ms, 0 late markers** |
| cross-shape (T301/T003/T201/T701/T008/T004) | all non-empty, expected class markers present (`standard 10/deviant 2`, `eyes_open 3/eyes_closed 3`, `load_low 6/load_high 6`, `blink 5/rest 7`, `sham_a 5/sham_b 7`, `click 12`) |
| back-dated markers (T101/T108) | 8/8 class markers share the exact `stim_onset` t_mono |
| effective sample rate over 54 s | 250.02 Hz |
| headless (no DISPLAY, null audio) | valid session via the terminal backend |
| `--dry-run` | 0 samples, 49 markers, valid session |
| SIGTERM abort | valid session, `notes.md` marked ABORTED |
| microphone session | `has_audio: True`, 16 kHz, 8.77 s, −21.6 dBFS, `audio_start`/`audio_stop` markers present |

## 6. Voice channel

`voice.py` is verified against synthetic ground truth, not just "it ran":

| injected | recovered |
|---|---|
| speech onset 1.5 s, cue at 1.0 s → 500 ms latency | **500.0 ms** |
| speech duration 2000 ms | **2000.0 ms** |
| one 300 ms internal gap | **1 pause** |
| envelope peaks at 4.5 /s | **4.65 syl/s** |
| silence | `spoke: False`, no false onset |

Live microphone probe: 2.016 s captured at 16 kHz, RMS −21.3 dBFS, peak 0.226,
"signal present" (the sample is discarded by default). A three-sentence
synthetic "reading aloud" recording analysed standalone gave onset 410 ms,
duration 7140 ms, 4.62 syl/s, 2 pauses — matching what was injected.

## 7. Agent bridge

From the agent workstream's own verification, re-checked with the integrator's
external-protocol assertions:

* `/health`, `/schema` (15 fields, 6 tool definitions), `/state`, `/metrics`,
  `/candidate` all answer with no hardware.
* `/candidate` returns `candidate_label: "uncertain"`,
  `probabilities_calibrated: false`, `model_version: null`,
  `class_probabilities: null`, with explicit `uncertain_reasons`.
* `POST /confirm` writes the participant's correction as a **separate** event
  with `reported_label`; the prediction was already durably logged with
  `predicted_label` and is never overwritten (22 predictions + 4 corrections
  coexisting in `/events`).
* Injected blink and clench artefacts are detected with confidence 1.0 at the
  right times, after fixing two real detector bugs (filter edge transients, and
  clench width measured on an oscillation peak).

## 8. What is **not** verified

Being precise about the gaps:

1. **No live headband session has been run.** Every acquisition test above used
   your already-recorded packets. The BLE handshake path in `device/th21a.py`
   is a faithful port of the code that produced those packets, but it has not
   been exercised against the device from NeuroBridge.
2. **The full 64-test sweep never finished at full decoding scale.** The
   pre-registered permutation test (200 permutations × the spec's CV) costs tens
   of seconds per test, and this machine was heavily contended by concurrent
   verification runs — the load average peaked around **60 on 8 cores**, because
   killing a parallel sweep left its worker processes orphaned and still
   computing. §4 shows the representative set that did complete at full
   fidelity, chosen to span every design shape: trial/ERP, condition/windowed,
   phase, and control.

   **Coverage was obtained separately**, in a pass that stubbed only
   `permutation_test` (and the false-positive-rate helper) and ran every other
   path: **64/64 specs evaluated, 0 signal-only guard failures, 0 exceptions**,
   with verdicts FAIL 47 / PASS 14 / INSUFFICIENT DATA 3 (T106, T501, T705 — too
   few epochs at smoke scale, correctly reported rather than guessed). No
   single-class spec ever invented an accuracy. The large FAIL count is expected
   at smoke scale with a stubbed permutation test; it is the *chance-level
   control* passing, not a defect. Reproduce with:

   ```bash
   .venv/bin/python scripts/verify_all.py --scale 0.03 --jobs 6 \
       --no-permutation --max-trials 100
   ```

   An `--no-permutation` pass answers "does the whole chain run", never "does
   this decode": p-values and the reported accuracies come from the full-fidelity
   passes in §4 and §5.

3. **`tests/test_smoke.py` was not run to completion.** pytest in this
   environment is broken by an incompatible `anyio` plugin
   (`ModuleNotFoundError: _pytest.scope`); with plugin autoload disabled the
   suite started but was killed under the same CPU contention. Its assertions
   were each exercised directly instead: catalog 64/64 valid, fake-source shapes
   and ERP injection, store round trip, `simulate`→decode, the `--effect-size 0`
   null model (0.4947–0.500), intent detection, and all bridge routes. The
   pipeline workstream reports the equivalent assertions passing:
   T301 0.9365 ≥ 0.62, T201 1.0000 ≥ 0.55, null model 0.4947 < 0.70.
4. **ERP latencies are unverified in absolute terms.** No round-trip latency
   measurement has been taken on this machine yet; run
   `voice.measure_roundtrip()` and `T004` before trusting any latency figure.
   Imported captures are tagged `timing_quality: coarse` for exactly this reason.
5. **T001's synthetic contact verdict is a stand-in.** The fake source now emits
   a four-`OK` self-check payload so the `good_contact` path is exercised, but
   real contact quality can only be judged on the device.

## 9. Session presentation fixes (fullscreen, no auto-advance, logs)

Reported by the user after their first session.

**Fullscreen was never actually full screen.** Three separate faults:

| check | before | after |
|---|---|---|
| `settings.fullscreen` default | `False`, so `RunConfig.fullscreen=None` → windowed | `True`; `--windowed` opts out |
| GUI "Fullscreen stimuli" checkbox | passed no flag at all — a no-op | passes `--fullscreen` / `--windowed` |
| `set_mode` size | hard-coded `1280x800` | the real desktop size |

Measured on the operator's desktop (1920x1080), which is why the bug looked the
way it did:

```
set_mode((1280, 800), FULLSCREEN)  -> WM geometry w=1280 h=800   <-- the bug
set_mode((1920,1080), FULLSCREEN)  -> WM geometry w=1920 h=1080  <-- the fix
```

A "fullscreen" session was a 1280x800 window in the corner of a 1920x1080
screen, with the desktop and the bottom panel visible around it. Verified after
the fix:

```
$ wmctrl -lG        # during a live session
0x05c00004  0 0  0  1920 1080  ...  NeuroBridge
$ nb screen-test
desktop size(s) : [(1920, 1080)]
presentation    : (1920, 1080)  driver=x11
OK covers a whole desktop ((1920, 1080))
```

Also now in place: the SDL env the child probe proves works
(`SDL_VIDEO_X11_FORCE_EGL=1`) is applied to the real session too; the cursor is
hidden and restored; a `WINDOWFOCUSLOST` fullscreen re-assert is deferred and
bounded so it can never stall a stimulus; and the **screensaver/display blanking
is inhibited** for the session (`systemd-inhibit (idle:sleep)` here, with D-Bus
and `xdg-screensaver` rungs above it) and released on every exit path. The dark
surface stays up through instructions, cues and breaks, so the desktop is never
exposed mid-session. `nb screen-test` exists so this is checkable in five seconds
instead of by wearing the headband.

**Sessions advanced by themselves.** `wait(duration, until=key_pressed)` returned
on whichever came first, and the countdown implied it would move on. Now:

* `--advance key` (default) waits for SPACE/RETURN on start, instructions,
  per-block cues, breaks and end — with a 1800 s safety cap only, logged as
  `advance_cap_hit`, and no countdown on screen;
* `--advance timer` keeps the old behaviour for rehearsal;
* `dry_run` forces timer, and a run with **no display and no TTY** downgrades to
  timer automatically, printing the reason and recording `advance_effective` +
  `advance_reason`, so automation cannot hang.

Verified through the real SDL event queue (posting `KEYDOWN`, since synthetic
`xdotool` events do not reach SDL on this host): 6 operator waits, session
completed in 12.3 s, cap not hit; and the non-interactive path prints
`advance: timer (no interactive input ...)` with the matching metadata.

**Two further crashes found and fixed while doing this**, both of which would
have hit the GUI: the old fullscreen re-assert called `set_mode` from inside the
event loop and produced measured stalls of 1.27 s / 3.96 s / 11.69 s; and
re-initialising pygame per presenter **segfaulted SDL** under fullscreen EGL on
the second window, so repeated in-process sessions (exactly what the control
panel does) would have died. SDL is now a once-only process-wide session.

**Focus trap, and what the GUI does about it.** A session waits for SPACE in its
*own* fullscreen window. If the control panel keeps keyboard focus, that SPACE
never arrives and the session sits at "Ready to begin" indefinitely. The panel now
minimises itself when it starts a session (and restores itself when the session
ends), prints the hint before doing so, and a watchdog notices 90 s of silence
during a session and says what to do. Verified: a session command minimises, a
non-session command does not, and the panel is restored on completion.

**Logs for reporting problems.** Three separate files, all additive:

| file | content |
|---|---|
| `data/logs/commands-YYYYMMDD.log` | every control-panel command, with a header (settings, device, backends, display) and per-line timestamps |
| `data/sessions/<id>/run.log` | everything a session printed, header + `[ elapsed s]` lines, written on abort too |
| `data/logs/diagnostic-*.txt` | one shareable bundle |

```
$ ./bin/nb diagnose --note "test of the shareable bundle"
OK wrote data/logs/diagnostic-20261013-154858.txt  (5412 bytes)
```

130 lines: host/os/python, display environment, package versions, settings with
**credentials withheld** (tokens reported `present`/`absent`, asserted), catalog
validation, the full doctor output, session summaries newest-first, model/report
counts, recent log tails and any tracebacks found in them. Collection is
read-only — it writes the one report and never modifies session data. In the GUI:
`Save console log…`, `Copy console log`, `Collect diagnostics…` (writes the file
and puts its path on the clipboard), and `Copy last session's run.log`; the
clipboard round-trip and the absence of credential material were both checked.

**One more bug, found by the runner's own testing:** `voice.MicRecorder.stop()`
reported impossible levels (an earlier session said `rms_dbfs: Inf`, this one
`+125.1 dBFS`, while the written WAV was merely clipped at about −3 dBFS) because
a device glitch can deliver NaN/Inf samples and the RMS was accumulated in
float32. Non-finite samples are now counted, reported as
`nonfinite_samples`/`nonfinite_pct` and zeroed in the returned copy, the mean is
computed in float64, and the reported level is clamped to digital full scale
(0 dBFS, with `level_capped` set). Verified with a deliberately glitched buffer:
3 non-finite samples counted, level `−20.8 dBFS` instead of `+125`.

## 10. Desktop integration (system menu)

```
$ ./scripts/install_desktop.sh
  installed icons: 256/128/64/48 px
  wrote ~/.local/share/applications/neurobridge.desktop
  wrote ~/.local/share/applications/neurobridge-bridge.desktop
  validated neurobridge.desktop
  validated neurobridge-bridge.desktop
$ desktop-file-validate ~/.local/share/applications/neurobridge.desktop
(no output = valid)
$ gtk-launch neurobridge
$ wmctrl -lx | grep -i neuro
0x0360004a  0 neuroBridge.Neurobridge  N/A  NeuroBridge — EEG training control panel
```

Both entries launch: the control panel opens (`Terminal=false`, its own Tk
window), and the bridge entry opens a terminal running `nb serve`. WM_CLASS
matches `StartupWMClass`, and the icons are installed into hicolor so the menu
and taskbar show the NeuroBridge icon.

**A real bug this caught.** `gtk-launch` runs with `$HOME` as the working
directory, and the wrappers were relying on the shell's cwd to make the
`neurobridge` package importable — so the menu entry died with
`No module named neurobridge.__main__` while every command-line invocation from
the project directory worked. Fixed two ways: the wrappers now `cd` to the
project root, and a `.pth` file in the venv's site-packages makes the package
importable from anywhere (`python -m neurobridge --version` verified from `/tmp`).

## 11. Re-running any of this

```bash
cd /path/to/neurobridge

./bin/nb doctor --scan --mic              # environment, headband, microphone
./bin/nb catalog validate                 # 64 specs
./bin/nb catalog audit                    # structural checks + duration disagreements
./bin/nb simulate T301 --scale 0.25       # end-to-end, decoding
./bin/nb simulate T301 --scale 0.25 --effect-size 0   # must land at chance
.venv/bin/python scripts/verify_all.py --tests T301 T201 T003 --scale 0.06 --jobs 3
.venv/bin/python scripts/verify_all.py --scale 0.03 --jobs 6 --no-permutation --max-trials 100
.venv/bin/python scripts/import_ble_packets.py ../ble/session_packets.txt --test-id T003
.venv/bin/python scripts/import_protocol.py
./scripts/install_desktop.sh               # (re)install the menu entries
./bin/nb gui                               # the control panel itself
```

Note on the sweep: run it on an idle machine, one pass at a time. Concurrent
passes on this box inflated the load average to ~60 and made every timing
number in this document take minutes instead of seconds.
