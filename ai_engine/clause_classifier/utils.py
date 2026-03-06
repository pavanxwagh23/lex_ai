"""
utils.py – Shared utilities for the Clause Classifier module.

Provides:
- Device auto-detection  (CUDA / MPS / CPU)
- HF Trainer metrics callback
- Canonical prediction dict formatter
- sklearn classification report printer
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Detect the best available compute device.

    Priority order: CUDA GPU → Apple MPS → CPU.

    Returns
    -------
    str
        PyTorch device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info("Using device: %s", device)
        return device

    except ImportError:
        logger.warning("torch not installed – defaulting to cpu")
        return "cpu"


# ---------------------------------------------------------------------------
# Hugging Face Trainer metrics callback
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Compute evaluation metrics inside the Hugging Face ``Trainer`` loop.

    Called automatically by ``Trainer.evaluate()`` after each epoch.

    Parameters
    ----------
    eval_pred : EvalPrediction
        Named tuple with fields ``predictions`` (logits, shape
        ``[N, num_labels]``) and ``label_ids`` (int array, shape ``[N]``).

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``f1``, ``precision``, ``recall``.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy  = accuracy_score(labels, predictions)
    f1        = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall    = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy":  round(accuracy, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


# ---------------------------------------------------------------------------
# Prediction output formatter
# ---------------------------------------------------------------------------

def format_prediction(label: str, score: float) -> dict[str, Any]:
    """
    Build the canonical prediction dict returned to callers and the API.

    Parameters
    ----------
    label : str
        Predicted clause label, e.g. ``"termination"``.
    score : float
        Softmax confidence score in ``[0, 1]``.

    Returns
    -------
    dict
        ``{"clause_type": str, "confidence": float}``
    """
    return {
        "clause_type": label,
        "confidence":  round(float(score), 4),
    }


# ---------------------------------------------------------------------------
# Classification report printer
# ---------------------------------------------------------------------------

def log_classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> None:
    """
    Print a full sklearn classification report to the logger at INFO level.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth label indices.
    y_pred : list[int]
        Predicted label indices.
    label_names : list[str]
        Human-readable label strings (must align with index ordering).
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
    logger.info("\n=== Classification Report ===\n%s", report)


# ---------------------------------------------------------------------------
# Tokenizer model name resolver
# ---------------------------------------------------------------------------

def resolve_model_name(primary: str, fallback: str) -> str:
    """
    Attempt to import and load the primary model name from Hugging Face Hub.

    If the primary model raises any error during tokenizer loading it falls
    back to ``fallback`` automatically — useful for air-gapped environments.

    Parameters
    ----------
    primary : str
        Preferred model identifier (e.g. ``"nlpaueb/legal-bert-base-uncased"``).
    fallback : str
        Safety net model identifier (e.g. ``"bert-base-uncased"``).

    Returns
    -------
    str
        The model name that should be used for subsequent loading calls.
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(primary)
        logger.info("Resolved primary model: %s", primary)
        return primary

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Primary model '%s' unavailable (%s). Falling back to '%s'.",
            primary,
            exc,
            fallback,
        )
        return fallback
