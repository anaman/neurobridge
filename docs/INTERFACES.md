# Module contracts

Frozen interfaces. Downstream modules import exactly these names; if you need
an extra helper, add it, but never rename or re-signature anything listed here
without updating every caller listed under "used by".

Conventions for every module in this project:

* Python 3.10, `from __future__ import annotations`, full type hints.
* No network or hardware access at import time. Import must succeed on a
  machine with no headband, no token and no internet.
* Never raise for a *missing optional resource* (token, ffmpeg, display,
  audio device): degrade, and report what happened in a `warnings` list.
* Docstrings explain **why** a design choice was made, not just what the code
  does. This is research code that will be read again in six months.
* Numerical code must be testable offline: no `time.sleep` in analysis paths.

---

## 1. `neurobridge/stimuli/` — content acquisition

Files: `cache.py`, `openverse.py`, `freesound.py`, `synth.py`, `textbank.py`,
`builtin.py`, `builder.py`, `__init__.py`.

`base.py` already defines `Stimulus`, `StimulusSet`, `prepare()`, `kind_dir()`.
`prepare()` delegates to `builder.build()`.

```python
# cache.py
@dataclass
class CachedFile:
    sha1: str; kind: str; path: Path; source: str          # openverse|freesound|synth|textbank|builtin
    url: str = ""; license: str = ""; attribution: str = ""
    title: str = ""; query: str = ""; meta: dict = field(default_factory=dict)

class StimulusCache:
    def __init__(self, root: Path | None = None) -> None            # default data/cache/stimuli
    def has(self, key: str) -> CachedFile | None
    def put_bytes(self, data: bytes, *, kind: str, suffix: str, source: str,
                  url: str = "", license: str = "", attribution: str = "",
                  title: str = "", query: str = "", meta: dict | None = None) -> CachedFile
    def put_path(self, src: Path, **kw) -> CachedFile               # copy/move into the cache
    def index(self) -> list[CachedFile]                            # reads cache/index.jsonl
    def add_attribution(self, cf: CachedFile) -> None              # appends to data/stimuli/ATTRIBUTION.md, deduped
def cache_key(**parts) -> str                                      # stable sha1 of sorted key=value pairs

# openverse.py   https://api.openverse.org/v1/  (anonymous works; OAuth2 optional)
class NetworkError(RuntimeError): ...
class NoResultsError(RuntimeError): ...
class OpenverseClient:
    BASE = "https://api.openverse.org/v1"
    def __init__(self, client_id: str = "", client_secret: str = "",
                 timeout: float = 30.0, user_agent: str = "NeuroBridge/0.1") -> None
    def search(self, query: str, n: int = 20, *, license_type: str = "all-cc",
               size: str | None = None, mature: bool = False,
               page_size: int = 20) -> list[dict]
    def download(self, item: dict, dest: Path) -> Path              # verify with PIL, downscale to <=1024 px long edge
    def fetch(self, query: str, n: int, cache: StimulusCache, *, force: bool = False) -> list[CachedFile]
    def has_network(self) -> bool
Credential names: NB_OPENVERSE_CLIENT_ID / NB_OPENVERSE_CLIENT_SECRET.
Handle HTTP 429 with Retry-After (max 2 retries) and 5xx with one retry.

# freesound.py   https://freesound.org/apiv2/   (token REQUIRED, even to search)
class NoTokenError(RuntimeError): ...
class AuthError(RuntimeError): ...
class FreesoundClient:
    BASE = "https://freesound.org/apiv2"
    def __init__(self, token: str = "", timeout: float = 30.0) -> None
    @property
    def available(self) -> bool
    def search(self, query: str, n: int = 10, *, min_duration: float | None = None,
               max_duration: float | None = None, license: str | None = None,
               sort: str = "score") -> list[dict]
    def download(self, item: dict, dest: Path, *, prefer: str = "preview_hq") -> Path
    def fetch(self, query: str, n: int, cache: StimulusCache, *, force: bool = False) -> list[CachedFile]
Credential name: NB_FREESOUND_TOKEN (also readable from neurobridge.yaml).
Previews are mp3: convert to 44.1 kHz mono WAV with ffmpeg when it is on PATH,
otherwise keep the mp3 and note it (libsndfile >= 1.2 can read mp3).

# synth.py   deterministic, dependency-light audio generation
def tone(freq: float, dur: float, fs: int = 44100, amp: float = 0.3, ramp_ms: float = 8.0) -> np.ndarray
def click(fs: int = 44100, dur: float = 0.004) -> np.ndarray
def click_train(n: int = 60, isi: float = 1.2, fs: int = 44100) -> tuple[np.ndarray, list[float]]
def noise(dur: float, kind: str = "pink", fs: int = 44100, amp: float = 0.25) -> np.ndarray
def chord(freqs: Sequence[float], dur: float, fs: int = 44100, *, dissonant: bool = False) -> np.ndarray
def melody(notes: Sequence[float], note_dur: float, fs: int = 44100) -> np.ndarray
def syllable(vowel: str = "a", consonant: str = "b", dur: float = 0.25,
             f0: float = 110.0, fs: int = 44100) -> np.ndarray      # formant synthesis
def amplitude_modulate(carrier_hz: float, mod_hz: float, depth: float, dur: float, fs: int = 44100) -> np.ndarray
def pink_noise(dur: float, fs: int = 44100) -> np.ndarray
def normalize_rms(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray
def save_wav(path: Path, x: np.ndarray, fs: int = 44100) -> Path
def generate(name: str, **kw) -> Path            # named generators, cached under data/stimuli/audio/synth/
NAMES: dict[str, Callable]                        # 'tone_500','tone_1000','tone_1500','click_train',
                                                  # 'syllable_ba','syllable_da','syllable_pa','noise_pink',
                                                  # 'chord_consonant','chord_dissonant','melody_familiar',
                                                  # 'melody_unfamiliar','am_10hz','beat_60','beat_120',
                                                  # 'pseudo_speech', ...

# textbank.py   public-domain / self-authored text, no copyrighted sources
@dataclass
class TextItem: text: str; label: str = ""; meta: dict = field(default_factory=dict)
def categories() -> list[str]
def items(category: str, n: int | None = None, seed: int = 20261013) -> list[TextItem]
def pseudowords(n: int, seed: int = 20261013) -> list[str]
def ensure_textbank(force: bool = False) -> Path          # writes data/stimuli/text/textbank.json
REQUIRED_CATEGORIES = ["simple_sentences", "garden_path_sentences", "n400_congruent",
    "n400_incongruent", "words_high_freq", "words_low_freq", "pseudowords",
    "passages_easy", "passages_medium", "passages_hard", "passages_positive",
    "passages_negative", "passages_neutral", "prose_passages", "shuffled_word_lists",
    "digits", "stroop_congruent", "stroop_incongruent", "nback_letters",
    "arithmetic_easy", "arithmetic_hard", "own_name", "other_names",
    "names_common", "questions_yes_no", "read_aloud_sentences"]

# builtin.py   procedurally drawn fallback images so image tests run offline
def ensure_builtin_images(force: bool = False) -> dict[str, list[Path]]
CATEGORIES = ["bright", "dark", "natural", "manmade", "face", "object",
              "pleasant", "neutral", "unpleasant", "standard", "deviant"]

# builder.py
def build(spec, allow_download: bool = True, force: bool = False,
          verbose: bool = True, limit: int | None = None) -> StimulusSet
Resolution rules:
  source none|internal  -> empty set, notes explaining the test is device-only
  kind text             -> textbank.items(category) per class (queries map label->category name or list)
  source synth          -> synth.generate(name) per class; `queries[label]` are generator names
  source openverse      -> cache first; on failure fall back to builtin.ensure_builtin_images()
  source freesound      -> cache first; on failure fall back to synth (list of fallback generator names)
  source mixed          -> openverse/freesound per class key suffix, else builtin/synth
Always: honour `n_per_class`, respect `max_items_per_test` from settings, populate
`StimulusSet.sources` counts, append a warning for every fallback used, and register
attribution rows for downloaded items.
```

---

## 2. `neurobridge/pipeline/` — analysis

Files: `preprocess.py`, `epochs.py`, `features.py`, `decoders.py`,
`evaluate.py`, `report.py`, `__init__.py`.

```python
# preprocess.py
def filter_signal(values, fs, bandpass=(0.3, 45.0), notch=(50.0, 60.0)) -> np.ndarray
def detect_artifacts(values, fs, ptp_limit=None) -> dict     # saturation_pct, motion_pct, flatline_pct, bad_frac
def artifact_mask(values, fs, ptp_limit) -> np.ndarray       # bool, True == bad sample
@dataclass
class PreprocessResult: signal: np.ndarray; fs: float; quality: QualityReport; artifacts: dict; meta: dict
def preprocess(values, fs, spec=None, **kw) -> np.ndarray
def preprocess_session(session, spec=None) -> PreprocessResult

# epochs.py
@dataclass
class Epochs:
    X: np.ndarray          # (n_epochs, n_samples) float32
    y: np.ndarray          # (n_epochs,) int class index
    labels: list[str]      # index -> label name
    groups: np.ndarray     # (n_epochs,) int block id, for grouped CV
    times: np.ndarray      # (n_samples,) seconds relative to onset
    fs: float
    onsets: np.ndarray     # (n_epochs,) monotonic onset time
    meta: dict
    def __len__(self) -> int
    def class_counts(self) -> dict[str, int]
    def balance(self) -> "Epochs"                       # subsample to the smallest class
def epochs_from_arrays(values, fs, onsets, tmin, tmax, baseline=None) -> np.ndarray
def epochs_from_session(session, spec, signal=None, event_name="stim_onset",
                        reject=True) -> Epochs           # label from spec.label_map()
def windowed_epochs(session, spec, window_s=2.0, step_s=0.5,
                    label_from="state") -> Epochs      # sustained-state tests: label from the
                                                       # `state` meta of the most recent class/phase marker
# features.py
FEATURE_NAMES: list[str]
def band_powers(values, fs) -> dict[str, float]
def extract(values, fs, features=None) -> dict[str, float]          # one window -> named features
def feature_names(features) -> list[str]
def feature_vector(feats, names=None) -> np.ndarray
def extract_epochs(epochs, spec) -> tuple[np.ndarray, list[str]]    # (n_epochs, n_features)
Implements exactly the vocabulary in docs/CATALOG_SPEC.md. Unknown feature
strings raise ValueError with the supported list.

# decoders.py
@dataclass
class TrainedModel:
    spec_id: str; name: str; feature_names: list[str]; classes: list[str]
    pipeline: Any; meta: dict
    def predict(self, X) -> np.ndarray
    def predict_proba(self, X) -> np.ndarray
def train(spec, X, y, groups=None, feature_names=None, seed=None) -> TrainedModel
def save_model(model, path=None) -> Path        # joblib/pickle under data/models/<spec_id>.joblib
def load_model(spec_id_or_path) -> TrainedModel
def model_path(spec_id) -> Path
class RiemannianTangentSpace:                   # sklearn-compatible, scipy logm/expm, works for 1 channel
class EEGNet(nn.Module):                        # 1-channel adaptation, importable without torch installed
# decoders must never crash on a single-class or tiny dataset: return a
# TrainedModel whose meta records why training was skipped.

# evaluate.py
def cross_validate(spec, X, y, groups=None, model=None) -> dict
    # -> {'folds':[...], 'balanced_accuracy':float, 'auc':float|None, 'f1':float,
    #     'confusion':[[...]], 'chance':float, 'n':int, 'notes':[...]}
def permutation_test(spec, X, y, groups=None, n=200, seed=0) -> dict   # {'p_value','observed','null_mean'}
def information_transfer_rate(accuracy, n_classes, trial_s) -> float   # bits/min (Wolpaw)
def apply_gates(spec, results) -> dict          # {'passed':bool,'checks':[{'metric','value','gate','ok'}]}
def evaluate_session(session, spec, model=None, train_if_needed=True) -> dict
def train_and_evaluate(session_paths, spec, save=True) -> dict

# report.py
def write_test_report(spec, results, session_path=None, out_dir=None) -> Path   # .md next to figures
def plot_erp(epochs, spec, out=None) -> Path
def plot_bandpower(epochs_or_feats, spec, out=None) -> Path
def plot_confusion(results, spec, out=None) -> Path
def plot_quality(session, out=None) -> Path
def plot_timeline(session, out=None) -> Path
def write_session_report(session, results_by_test) -> Path
def write_curriculum_report(results_by_test) -> Path     # the 64-test progress table
All plots use matplotlib 'Agg' and MPLCONFIGDIR under data/cache/matplotlib.
```

---

## 3. `neurobridge/runner/` — presentation & acquisition

Files: `present_audio.py`, `present_visual.py`, `timeline.py`, `session.py`,
`__init__.py`.

```python
# present_audio.py
class AudioPresenter:
    def __init__(self, backend: str = "auto", fs_out: int = 44100, gain: float = 1.0) -> None
    @property
    def backend(self) -> str                      # sounddevice | aplay | null
    @property
    def latency_ms(self) -> float                 # nominal output latency for this backend
    def play(self, path, *, blocking: bool = False) -> float    # returns onset t_mono
    def play_array(self, x: np.ndarray, fs: int, *, blocking: bool = False) -> float
    def play_tone(self, freq: float, dur_s: float, *, blocking: bool = False) -> float
    def stop(self) -> None
    def close(self) -> None
# present_visual.py
class VisualPresenter:
    def __init__(self, size=(1280, 800), fullscreen: bool = False,
                 backend: str = "auto", bg=(18, 18, 20)) -> None
    @property
    def active(self) -> bool
    def show_text(self, text, *, font_size=30, duration_s=None, color=(235,235,235)) -> float
    def show_image(self, path, *, duration_s=None, max_frac=0.8) -> float
    def show_fixation(self, char="+", duration_s=None) -> float
    def show_blank(self, duration_s=None) -> float
    def flash(self, color=(255,255,255), duration_s=0.05) -> float
    def show_countdown(self, seconds: float, label: str = "") -> None
    def show_progress(self, frac: float, label: str = "") -> float
    def poll_events(self) -> list[dict]           # [{'type':'key','key':'space','t':...},{'type':'quit'}]
    def pump(self, seconds: float, *, poll_cb=None) -> list[dict]   # display + poll for `seconds`, calling poll_cb()
    def close(self) -> None
All timestamps are `time.perf_counter()` taken *after* the frame flip / audio start.
`pump()` is how the runner keeps the BLE acquisition thread drained while waiting.

# timeline.py
@dataclass
class Trial: index: int; block: int; label: str; stim: Stimulus | None
             onset_code: int; meta: dict = field(default_factory=dict)
@dataclass
class Phase: kind: str; label: str; trials: list[Trial] = ...; duration_s: float | None = None
def build_trials(spec, stimset, rng=None) -> list[Trial]
def make_phases(spec, trials, include_practice=True) -> list[Phase]
def jitter(rng, lo, hi) -> float

# session.py
@dataclass
class RunConfig:
    spec: TestSpec; device: str = "auto"; session_id: str | None = None
    no_device: bool = False; dry_run: bool = False; duration_scale: float = 1.0
    fullscreen: bool | None = None; verbose: bool = True; limit: int | None = None
    allow_download: bool = True; seed: int | None = None
def run_test(config: RunConfig) -> Path
def run_sequence(spec_ids: list[str], **kw) -> Path      # one session, several tests, one BLE connection
class Runner:                                           # public enough to drive from nb serve
    def __init__(self, config: RunConfig) -> None
    def setup(self) -> None
    def run(self) -> Path
    def teardown(self) -> None
    @property
    def session_path(self) -> Path | None
```

Runner requirements:

1. One `EEGSource`, one `SessionWriter`, one `MultiMarkerSink`
   (`SessionMarkerSink` + optional `LSLMarkerSink`) plus `source.notify_marker`
   for every marker.
2. All waiting goes through an internal `wait(seconds)` that polls the device
   every ≤5 ms and feeds `SessionWriter.add_samples`, so acquisition never
   stalls during a 30 s break.
3. Every presented stimulus logs its code with the timestamp returned by the
   presenter (`stim_onset`), and also logs `stim_offset`.
4. Keyboard input is read via `VisualPresenter.poll_events()` (pygame) and is
   entirely optional: with `--no-display` the runner prints instructions and
   auto-advances after a countdown.
5. Abort on `q`/ESC writes remaining data, closes the device, and still leaves
   a valid session directory with `notes.md` saying the session was aborted.
6. `dry_run=True` presents nothing: it exercises the marker/timing path and
   writes a session with markers but no samples (for testing without a device
   or a screen).
7. After the run: quality assessment, timing report, and a one-screen summary.
8. `duration_scale` scales every delay so a full test can be smoke-tested in
   ~60 s.

---

## 4. `neurobridge/agent/` — the brain-to-agent bridge

Files: `state.py`, `bridge.py`, `link.py`, `protocol.md`, `__init__.py`.

```python
# state.py
@dataclass
class StateEstimate:
    t_mono: float; t_wall: str
    attention: float; arousal: float; load: float; fatigue: float
    relaxation: float; valence: float | None
    artifact_level: float; contact: str; confidence: float
    source: str; intents: list[dict]; model_info: dict
    def as_dict(self) -> dict
class StateEstimator:
    def __init__(self, fs=250.0, window_s=4.0, step_s=0.5, models_dir=None,
                 device="auto", replay_session=None) -> None
    def push(self, values, t_mono) -> None
    def push_packets(self, packets) -> None
    def estimate(self) -> StateEstimate          # cheap: reuses the current window
    def calibrate(self) -> dict                  # first 60 s of data == personal baseline
    def reset(self) -> None
    @property
    def ready(self) -> bool
class IntentDetector:
    def __init__(self, fs=250.0, blink_sigma=6.0, clench_band=(20, 45), refractory_s=0.6) -> None
    def push(self, values, t_mono) -> list[dict]  # [{'intent':'blink'|'clench','t':float,'confidence':float}]
Derived-state definitions must be documented in the docstrings and must be
computed from *relative* band power (the device is uncalibrated), using a
personal baseline captured at session start. Nothing here may claim to read
words or sentences: the honest ceiling is documented in protocol.md.

# bridge.py
def create_app(estimator=None, device="auto") -> "fastapi.FastAPI"
def serve(host=None, port=None, device="auto", replay_session=None) -> None
Routes:
  GET  /health          -> {'ok':bool,'device':str,'estimator_ready':bool,'sessions':int,'version':str}
  GET  /schema          -> self-describing JSON: every field, its range, meaning, and example tool descriptions
  GET  /state           -> current StateEstimate as JSON
  GET  /state/stream    -> text/event-stream (SSE), one event per state update
  GET  /samples/stream  -> SSE raw EEG counts for an attached recorder/display:
                           ?max_hz=&chunk_ms=&from_now=. Frames carry t0, fs, n,
                           stride, gap and values. max_hz decimates for drawing and
                           does NOT anti-alias, so a recorder takes the full rate
                           (the default). The bridge owns the BLE link, so this is
                           the only way a second process gets raw samples.
  GET  /events          -> marker/intent events, `?since=<t_mono>&limit=N`
  POST /mark            -> the agent can inject its own marker {'name':str,'meta':{}} into the session log
  WS   /ws              -> same payloads as /state/stream, plus inbound {'type':'mark'|'ping'}
  GET  /report/{test_id}-> latest evaluation JSON for a test
  GET  /candidate       -> the §10 candidate signal (always `uncertain` until a
                           validated decision rule exists)
  POST /confirm         -> the participant's own correction of a candidate
  POST /calibrate       -> capture the personal baseline now; 200 when a baseline
                           exists, 409 {'ok':false,'reason':'not_enough_data'}
                           while the bridge is still collecting. Nothing else in
                           the live path establishes one, so without a call here
                           every 0..1 field stays pinned at un-baselined 0.5.
The app must run with no hardware (synthetic source) and must never 500: on any
internal error return a JSON error object with the traceback in `detail`.
```
`protocol.md` is written for an LLM agent: what each field means, the polling
cadence, how to gate actions on `confidence`, which intents are safe to act on,
and an explicit "what this cannot do" section.

```python
# link.py -- the client half, for an agent that thinks in discrete turns
@dataclass(frozen=True)
class Provenance:                       # built from GET /health, never /state
    device: str; requested_device: str; mode: str
    is_synthetic: bool                  # generated: not a person, ever
    is_replay: bool                     # real EEG, wrong moment
    is_live_human: bool                 # the only case worth acting on
    label: str; banner: str
class BridgeClient:
    def __init__(self, base_url=None, timeout=8.0) -> None
    def health() -> dict; def provenance() -> Provenance
    def state(explain=False) -> dict; def schema() -> dict
    def candidate() -> dict; def metrics() -> dict
    def events(since=None, limit=100) -> dict
    def mark(name, meta=None) -> dict
    def confirm(candidate_seq, participant_response, note="") -> dict
    def calibrate(force=False) -> dict      # same payload even on 409: "wait" != "down"
    def stream(stop=None, on_open=None) -> Iterator[dict]   # adds _recv_mono/_recv_wall/_device
def probe_liveness(client, gap_s=0.7) -> dict               # two reads: is the clock advancing?
def record(client, out_path=None, seconds=None, ...) -> dict   # JSONL with a provenance header
def load_log(path, last_s=None) -> tuple[dict, list[dict]]
def summarize(rows, header=None, expected_hz=None) -> dict   # stats + computed caveats
def parse_duration("10m") -> float | None
```
Errors: `LinkError` (base), `BridgeUnreachable` (nothing listening),
`BridgeError` (answered with an error payload). `summarize` never reports a
field derived from `uncalibrated_neutral` as a measurement, and returns a
`limitations` list naming why *that* window cannot support a claim about a
person.

---

## 5. `neurobridge/monitor/` — the standalone monitor

Files: `sources.py`, `recorder.py`, `app.py`, `__init__.py`.

`app.py` is the only module that imports tkinter, so importing the package on a
headless machine stays safe. Nothing here opens hardware at import time.

```python
# sources.py
@dataclass
class Chunk:   values: np.ndarray; times: np.ndarray; gap: bool = False
@dataclass
class Snapshot:
    t_mono: float; t_wall: str; fields: dict[str,float]
    confidence: float; artifact_level: float; contact: str; source: str
    seq: int|None; informative: bool; derivation: str
    bands: dict[str,float]        # percent of 1-45 Hz, for display
    intents: list[dict]; calibrated: bool|None
@dataclass
class Reading: chunks: list[Chunk]; snapshot: Snapshot|None
class MonitorSource(abc.ABC):
    name: str; fs: float
    def open() -> None; def close() -> None
    def poll(timeout_s=0.02) -> Reading          # never blocks longer than timeout
    def calibrate(force=False) -> dict
    def provenance() -> Provenance
class DeviceSource(MonitorSource):   # owns the headband + a local StateEstimator
    def __init__(device="auto", state_hz=2.0, models_dir=None)
class BridgeSource(MonitorSource):   # attaches to nb serve: /state + /samples
    def __init__(url=None, state_hz=2.0, max_hz=250.0)
class ReplaySource(MonitorSource):   # a session folder or a monitor eeg.csv
    def __init__(path, state_hz=2.0, speed=1.0, models_dir=None, loop=False)
    def play(); def pause(); playing: bool; def set_speed(x); def seek(rel_s)
    def restart(); position -> (elapsed_s, duration_s)
def open_source_for_mode(device="auto", url=None, replay=None, ...) -> MonitorSource
BANDS_FOR_DISPLAY = ("delta","theta","alpha","beta","gamma")

# recorder.py
class MonitorRecorder:
    def __init__(out_dir=None, session_id=None, meta=None, provenance=None,
                 csv_enabled=True, native_enabled=True, flush_every=2500)
    def add_samples(values, times) -> None       # full rate, buffered
    def add_state(t_mono, t_wall, fields, *, confidence, artifact_level, contact,
                  source, seq, informative=True, derivation="", intents=None) -> None
    def event(name, meta=None, t_mono=None) -> float
    def close(summary=None) -> RecorderReceipt
@dataclass
class RecorderReceipt: path, session_id, n_samples, n_state_rows, n_events, ...
EEG_CSV_COLUMNS / STATE_CSV_COLUMNS / EVENT_CSV_COLUMNS

# app.py
@dataclass
class MonitorConfig: device, url, replay, record, out_dir, window_s, state_hz, max_hz
class MonitorApp:                                    # tkinter window
    def on_view_change()                             # 1/5/10/15/30/60 min trace
def run_monitor(config) -> int                       # opens the window, blocks
def run_headless(config, seconds=None, on_tick=None) -> RecorderReceipt|None
VIEW_CHOICES = (1, 5, 10, 15, 30, 60)                # minutes on the toolbar
RETAIN_MINUTES = 60                                  # history kept, whatever the view
```
The trace is drawn as a min/max envelope (one canvas line, ~3-10 ms per redraw even
for an hour of 250 Hz data). History is retained for the longest selectable view,
and `_visible()` walks the buffer backwards so a 1-minute view concatenates ~15k
samples rather than the full hour it sits in.
The recorder writes **both** full-rate CSV and the native session format from the
same calls, so a monitor recording is usable by `nb eval`/`nb train` and the two
cannot drift apart. `run_headless` uses the same source and recorder objects as the
window, so a headless capture is the same kind of recording rather than a second
code path.

---

## 6. Wiring (owned by the integrator, do not edit)

`neurobridge/cli.py` and `neurobridge/__main__.py` import the modules above.
`neurobridge/simulate.py` drives the fake device end-to-end.
