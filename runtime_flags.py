"""Runtime feature flags for evaluation and experiments."""

from __future__ import annotations

import os


def is_model_eval_mode() -> bool:
    return os.environ.get("MODEL_EVAL_MODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_minimal_mode() -> bool:
    return os.environ.get("MINIMAL_APP_MODE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

