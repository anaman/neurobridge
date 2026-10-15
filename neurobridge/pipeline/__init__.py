"""Analysis pipeline: preprocess -> epochs -> features -> decode -> evaluate -> report.

Re-exports the public names of the six modules so callers can write
``from neurobridge.pipeline import preprocess_session, extract_epochs, train, ...``
without knowing which file a function lives in.

Importing this package must never require torch, a display, or a writable home
directory: ``EEGNet`` is exposed lazily through the module ``__getattr__``
(PEP 562), and matplotlib is imported inside the report functions.
"""
from __future__ import annotations

from .decoders import (  # noqa: F401
    EEGNetClassifier,
    RiemannianTangentSpace,
    TrainedModel,
    build_pipeline,
    load_model,
    model_path,
    save_model,
    train,
)
from .epochs import Epochs, epochs_from_arrays, epochs_from_session, windowed_epochs  # noqa: F401
from .evaluate import (  # noqa: F401
    apply_gates,
    assert_signal_only,
    build_epochs,
    coverage_of,
    cross_validate,
    epoching_mode,
    evaluate_session,
    false_activations_per_min,
    information_transfer_rate,
    permutation_test,
    resolve_grouping,
    timing_quality_of,
    train_and_evaluate,
)
from .features import (  # noqa: F401
    BANDS,
    DEFAULT_FEATURES,
    FEATURE_NAMES,
    band_powers,
    extract,
    extract_epochs,
    feature_names,
    feature_vector,
)
from .preprocess import (  # noqa: F401
    PreprocessResult,
    artifact_mask,
    detect_artifacts,
    filter_signal,
    preprocess,
    preprocess_session,
)
from .report import (  # noqa: F401
    plot_bandpower,
    plot_confusion,
    plot_erp,
    plot_quality,
    plot_timeline,
    write_curriculum_report,
    write_session_report,
    write_test_report,
)

__all__ = [
    # preprocess
    "filter_signal", "detect_artifacts", "artifact_mask", "PreprocessResult",
    "preprocess", "preprocess_session",
    # epochs
    "Epochs", "epochs_from_arrays", "epochs_from_session", "windowed_epochs",
    # features
    "FEATURE_NAMES", "DEFAULT_FEATURES", "BANDS", "band_powers", "extract",
    "feature_names", "feature_vector", "extract_epochs",
    # decoders
    "TrainedModel", "train", "build_pipeline", "save_model", "load_model",
    "model_path", "RiemannianTangentSpace", "EEGNetClassifier", "EEGNet",
    # evaluate
    "cross_validate", "permutation_test", "information_transfer_rate",
    "apply_gates", "evaluate_session", "train_and_evaluate",
    "assert_signal_only", "timing_quality_of", "resolve_grouping", "epoching_mode",
    "build_epochs", "coverage_of", "false_activations_per_min",
    # report
    "write_test_report", "plot_erp", "plot_bandpower", "plot_confusion",
    "plot_quality", "plot_timeline", "write_session_report",
    "write_curriculum_report",
]


def __getattr__(name: str):
    """Expose ``EEGNet`` lazily so this package imports without torch."""
    if name == "EEGNet":
        from .decoders import EEGNet
        return EEGNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
