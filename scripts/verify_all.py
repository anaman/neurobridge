#!/usr/bin/env python3
"""Verify the whole 64-test system end to end, with no hardware.

    .venv/bin/python scripts/verify_all.py               # all 64, fast
    .venv/bin/python scripts/verify_all.py --scale 0.3   # longer synthetic blocks
    .venv/bin/python scripts/verify_all.py --tests T301 T201

For each test this:

1. builds a synthetic session from the spec (markers, blocks, jitter and all),
2. runs the ordinary analysis pipeline over it,
3. checks the pipeline's own pre-registered gates,
4. reports whether the result is *expected*, given what the synthetic source
   actually injects for that test's class markers.

The last point is what makes this a real test rather than a smoke test:

* if the synthetic source injects a known effect for a test and the pipeline
  stays at chance, the pipeline is broken;
* if the synthetic source injects **nothing** (the null model) and the pipeline
  still decodes well above chance, there is leakage somewhere;
* if the injected effect is real and the decoder finds it, the result is
  consistent -- and the same code path is what will run on the headband.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from neurobridge import store  # noqa: E402
from neurobridge.catalog import load_one, load_specs, validate_catalog  # noqa: E402
from neurobridge.config import REPORT_DIR, ensure_dirs  # noqa: E402
from neurobridge.device.fake import DEFAULT_POLICY, FakeSource  # noqa: E402
from neurobridge.pipeline import evaluate as E  # noqa: E402
from neurobridge.simulate import simulate_test  # noqa: E402


def synthetic_effect(spec) -> tuple[str, float]:
    """What does the synthetic source inject for this test's classes?

    Returns ``(kind, strength)`` where kind is one of:
      'erp'    - stimulus-locked responses that differ in amplitude
      'bands'  - sustained band-power states that differ
      'none'   - nothing: this test is a null model for the pipeline

    A class with **no** policy entry contributes zero, not "missing": T101's
    `target` markers get a large injected P300 while its `lapse` markers get
    nothing, which is a real injected difference. Treating the absent class as
    unknown made the harness call that decoding "leakage", which was wrong --
    the bug was here, not in the pipeline.
    """
    erps: dict[str, float] = {}
    bands: dict[str, float] = {}
    for cls in spec.classes:
        key = cls.lower().replace(" ", "_").replace("-", "_")
        pol = DEFAULT_POLICY.get(key) or {}
        if pol.get("kind") == "bands":
            bands[cls] = float((pol.get("bands") or {}).get("alpha", 1.0))
        elif pol.get("kind") in ("blink", "clench", "swallow"):
            erps[cls] = 250_000.0 if pol["kind"] == "blink" else 40_000.0
        else:
            erps[cls] = float(pol.get("amp", 0.0))      # absent class -> 0
        bands.setdefault(cls, float((pol.get("bands") or {}).get("alpha", 1.0)))
    if len(bands) >= 2 and (max(bands.values()) - min(bands.values())) > 0.15:
        return "bands", max(bands.values()) - min(bands.values())
    if len(erps) >= 2 and (max(erps.values()) - min(erps.values())) > 500:
        return "erp", max(erps.values()) - min(erps.values())
    return "none", 0.0


def effective_scale(spec, global_scale: float, min_cond_s: float = 12.0) -> tuple[float, str]:
    """Per-test duration scale, so windowed tests are not starved of windows.

    A single global scale is wrong for this catalog. Trial-based ERP tests keep
    their trial count whatever the scale (only the inter-trial interval
    compresses), so a fast scale is fine. Windowed/sustained tests cut analysis
    windows out of a *condition duration*, so compressing a 120 s condition to
    5 % leaves two windows and the test cannot be evaluated at all - which looks
    like a pipeline failure but is really a harness artefact.

    So: compress freely for trial designs, and for condition designs compress
    only as far as ``min_cond_s`` allows. Pass a small ``min_cond_s`` for a fast
    "does everything run" pass; the default is for a pass that actually tries to
    decode.
    """
    from neurobridge.simulate import condition_seconds, is_trial_based
    if is_trial_based(spec):
        return global_scale, "trial design: global scale"
    blocks = max(1, int((spec.design or {}).get("blocks", 1) or 1))
    cond = condition_seconds(spec, blocks)
    if cond <= 0:
        return global_scale, "no condition duration declared"
    if cond < min_cond_s:
        return 1.0, f"condition {cond:.0f}s already short: unscaled"
    s = max(global_scale, min(1.0, min_cond_s / cond))
    return s, f"condition {cond:.0f}s -> {s:.2f}x ({cond*s:.0f}s per condition)"



def _stub_permutation(*a, **kw) -> dict:
    """Stand-in for the real permutation test during a *coverage* pass.

    The real test is what makes an accuracy claim meaningful, so it is never
    stubbed in a decoding pass. It is stubbed here only to answer a different
    and much cheaper question -- "does the whole chain run on all 64 specs
    without an error?" -- because 200 permutations x 64 tests is hours.
    """
    return {"p_value": None, "observed": None, "null_mean": None, "null_std": None,
            "n": 0, "n_permutations": 0, "grouping": None,
            "notes": ["permutation test STUBBED (verify_all --no-permutation, coverage pass)"]}


def run_one(test_id: str, scale: float, seed: int, effect_size: float,
            keep: bool, min_cond_s: float = 12.0,
            no_permutation: bool = False, max_trials: int | None = None) -> dict:
    if no_permutation:
        import neurobridge.pipeline.evaluate as _E
        _E.permutation_test = _stub_permutation
    spec = load_one(test_id)
    used_scale, scale_why = effective_scale(spec, scale, min_cond_s)
    row: dict = {"id": spec.id, "title": spec.title, "tier": spec.tier,
                 "feasibility": spec.feasibility, "classes": spec.classes,
                 "status": "ok", "scale": round(used_scale, 3), "scale_why": scale_why}
    kind, strength = synthetic_effect(spec)
    row["injected"] = kind
    row["injected_strength"] = strength
    t0 = time.time()
    try:
        path = simulate_test(spec, effect_size=effect_size, seed=seed, scale=used_scale,
                             seed_note=f"verify_all scale={used_scale:.3f} seed={seed}",
                             max_trials=max_trials)
        session = store.Session.load(path)
        row["n_samples"] = int(session.eeg.size)
        row["duration_s"] = round(session.duration_s, 2)
        res = E.evaluate_session(session, spec)
        cv = res.get("cv", {}) or {}
        row["n_epochs"] = cv.get("n")
        row["balanced_accuracy"] = _finite(cv.get("balanced_accuracy"))
        row["chance"] = _finite(cv.get("chance"))
        row["auc"] = _finite(cv.get("auc"))
        row["f1"] = _finite(cv.get("f1"))
        try:
            row["itr_bits_per_min"] = res.get("itr_bits_per_min")
        except Exception:
            pass
        gates = E.apply_gates(spec, res)
        row["passed"] = bool(gates.get("passed"))
        row["checks"] = gates.get("checks", [])
        row["notes"] = (cv.get("notes") or [])[:3]
        row["session"] = str(path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        row["traceback"] = traceback.format_exc()[-1500:]
    row["elapsed_s"] = round(time.time() - t0, 1)
    if not keep and row.get("session"):
        import shutil
        shutil.rmtree(row["session"], ignore_errors=True)
    return row


def _finite(x):
    """Coerce nan/inf to None: an un-computable metric is not a failed one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def verdict_for(row: dict) -> str:
    """Judge the run against what was actually injected."""
    if row["status"] != "ok":
        return "ERROR"
    if len(row.get("classes") or []) < 2:
        # single-class commissioning tests (T004/T005/T006/T107...) have no
        # accuracy to judge; their gates are quality/SNR metrics instead
        return "single-class"
    acc, chance = _finite(row.get("balanced_accuracy")), _finite(row.get("chance"))
    if acc is None or chance is None:
        return "NO-DATA"
    lift = acc - chance
    injected = row["injected"]
    if injected == "none":
        # nothing was injected: a large lift means leakage
        return "LEAK?" if lift > 0.18 else "null-ok"
    if lift >= 0.10:
        return "decoded"
    if lift >= 0.04:
        return "weak"
    return "missed"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", nargs="*", default=None, help="subset of test ids")
    p.add_argument("--scale", type=float, default=0.08,
                   help="duration scale for the synthetic session (default: fast)")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--effect-size", type=float, default=1.0)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel test evaluations (each is independent; default 1)")
    p.add_argument("--min-cond-s", type=float, default=12.0,
                   help="minimum condition length for windowed tests (smaller = faster, "
                        "fewer analysis windows; 4 is a fast 'does it run' pass)")
    p.add_argument("--keep", action="store_true", help="keep the synthetic sessions")
    p.add_argument("--max-trials", type=int, default=None,
                   help="cap trials per test (verification only): a 1600-trial MMN "
                        "spec otherwise dominates the sweep")
    p.add_argument("--no-permutation", action="store_true",
                   help="stub the permutation test: cheap coverage pass, no p-values")
    p.add_argument("--json", default=None, help="write the full result JSON here")
    a = p.parse_args(argv)

    ensure_dirs()
    problems = validate_catalog()
    if problems:
        print("catalog validation problems:")
        for k, v in problems.items():
            print(f"  {k}: {v[:3]}")
        return 2

    specs = load_specs()
    if a.tests:
        want = {t.upper() for t in a.tests}
        specs = [s for s in specs if s.id in want or s.slug in a.tests]
    print(f"verifying {len(specs)} tests  (scale={a.scale}, seed={a.seed}, "
          f"effect_size={a.effect_size}, jobs={a.jobs})", flush=True)
    print("note: each eval includes the spec's pre-registered permutation test, so a "
          "test takes tens of seconds; that is analysis cost, not a hang.\n", flush=True)

    rows: list[dict] = []
    head = f"{'id':5s} {'inj':6s} {'epochs':>6s} {'acc':>6s} {'chance':>6s} {'lift':>6s} {'auc':>6s}  verdict"
    print(head, flush=True)
    print("-" * len(head), flush=True)

    def emit(row: dict) -> None:
        acc, ch, auc = row.get("balanced_accuracy"), row.get("chance"), row.get("auc")
        lift = None if (acc is None or ch is None) else acc - ch
        print(f"{row['id']:5s} {row['injected']:6s} {(row.get('n_epochs') or 0):6d} "
              f"{'-' if acc is None else f'{acc:6.3f}'} "
              f"{'-' if ch is None else f'{ch:6.3f}'} "
              f"{'-' if lift is None else f'{lift:+6.3f}'} "
              f"{'-' if auc is None else f'{auc:6.3f}'}  {row['verdict']}", flush=True)
        if row["status"] != "ok":
            print(f"      !! {row.get('error')}", flush=True)

    if a.jobs and a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(a.jobs)) as pool:
            futs = {pool.submit(run_one, sp.id, a.scale, a.seed, a.effect_size, a.keep,
                                a.min_cond_s, a.no_permutation, a.max_trials): sp.id
                    for sp in specs}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as exc:                      # worker died
                    row = {"id": futs[fut], "title": "", "tier": None,
                           "feasibility": "unknown", "classes": [],
                           "status": "error", "error": f"worker failed: {exc!r}",
                           "injected": "unknown", "verdict": "ERROR"}
                row["verdict"] = verdict_for(row)
                rows.append(row)
                emit(row)
    else:
        for spec in specs:
            row = run_one(spec.id, a.scale, a.seed, a.effect_size, a.keep,
                          a.min_cond_s, a.no_permutation, a.max_trials)
            row["verdict"] = verdict_for(row)
            rows.append(row)
            emit(row)
    rows.sort(key=lambda r: r["id"])


    # ---------------- summary
    def count(v: str) -> int:
        return sum(1 for r in rows if r["verdict"] == v)

    print("\nsummary")
    print(f"  decoded (lift >= 0.10)   {count('decoded'):3d}")
    print(f"  weak    (0.04 .. 0.10)   {count('weak'):3d}")
    print(f"  missed  (< 0.04)         {count('missed'):3d}")
    print(f"  null-ok (nothing injected){count('null-ok'):3d}")
    print(f"  single-class (no accuracy) {count('single-class'):3d}")
    print(f"  LEAK?   (nothing injected but decoded) {count('LEAK?'):3d}")
    print(f"  errors                   {count('ERROR'):3d}")
    print(f"  no-data                  {count('NO-DATA'):3d}")

    failures = []
    for r in rows:
        if r["verdict"] == "ERROR":
            failures.append(f"{r['id']}: pipeline error: {r.get('error')}")
        elif r["verdict"] == "LEAK?":
            failures.append(f"{r['id']}: nothing was injected yet it decoded at "
                            f"{r.get('balanced_accuracy'):.3f} - check for leakage")
        elif r["verdict"] == "missed" and r["feasibility"] in ("established", "promising"):
            failures.append(f"{r['id']}: injected effect ({r['injected']}) was not recovered "
                            f"(acc {r.get('balanced_accuracy')}) - pipeline or timeline bug")

    out = REPORT_DIR / "verify_all.json"
    out.write_text(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "scale": a.scale, "seed": a.seed,
                               "effect_size": a.effect_size, "rows": rows}, indent=2,
                              default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, default=str))

    # markdown companion
    md = ["# Synthetic end-to-end verification", "",
          f"*scale={a.scale}, seed={a.seed}, effect_size={a.effect_size}, "
          f"{time.strftime('%Y-%m-%d %H:%M')}*", "",
          "`injected` is what the synthetic source actually put into the signal for that "
          "test's class markers. `verdict` judges the pipeline against that, not against "
          "the test's own gate.", "",
          "| id | injected | epochs | bal-acc | chance | lift | auc | verdict | title |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc, ch = r.get("balanced_accuracy"), r.get("chance")
        lift = "-" if (acc is None or ch is None) else f"{acc-ch:+.3f}"
        md.append(f"| {r['id']} | {r['injected']} | {r.get('n_epochs') or '-'} | "
                  f"{'-' if acc is None else f'{acc:.3f}'} | "
                  f"{'-' if ch is None else f'{ch:.3f}'} | {lift} | "
                  f"{'-' if r.get('auc') is None else f'{r[chr(97)+chr(117)+chr(99)]:.3f}'} | "
                  f"{r['verdict']} | {r['title']} |")
    (REPORT_DIR / "verify_all.md").write_text("\n".join(md) + "\n")
    print(f"\nfull results: {out}\nmarkdown:     {REPORT_DIR / 'verify_all.md'}")

    if failures:
        print("\nFAILURES")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall expected effects recovered, no leakage detected, no pipeline errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
