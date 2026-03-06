"""
predict.py – Inference module for the Legal Clause Classifier.

Exposes three callables:

1. ``ClauseClassifier``        – reusable class that holds the loaded model
2. ``predict_clause``          – module-level function for single-paragraph calls
3. ``predict_batch_clauses``   – module-level function for batch predictions

FastAPI integration example
---------------------------
::

    from ai_engine.clause_classifier.predict import predict_clause

    @app.post("/classify_clauses")
    async def classify_endpoint(request: ClassifyRequest):
        return predict_clause(request.paragraph)

spaCy / transformers pipeline integration
-----------------------------------------
::

    from ai_engine.clause_classifier.predict import ClauseClassifier

    clf = ClauseClassifier()
    records = clf.predict_batch(paragraphs)

Requirements
------------
The model must have been trained and saved first::

    python ai_engine/clause_classifier/train.py
"""

import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make the package importable when run as a script
# ---------------------------------------------------------------------------
_CLAUSE_CLASSIFIER_DIR = Path(__file__).resolve().parent
_AI_ENGINE_DIR = _CLAUSE_CLASSIFIER_DIR.parent
_PROJECT_ROOT = _AI_ENGINE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_engine.clause_classifier.config import (
    CLAUSE_LABELS,
    FALLBACK_MODEL_NAME,
    ID2LABEL,
    INFERENCE_BATCH_SIZE,
    LABEL2ID,
    MAX_LENGTH,
    MIN_CONFIDENCE_THRESHOLD,
    MODEL_SAVE_DIR,
    NUM_LABELS,
    PRIMARY_MODEL_NAME,
)
from ai_engine.clause_classifier.utils import format_prediction, get_device

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton (loaded lazily on first call)
# ---------------------------------------------------------------------------
_classifier: "ClauseClassifier | None" = None


# ===========================================================================
# ClauseClassifier class
# ===========================================================================

class ClauseClassifier:
    """
    Inference wrapper around a fine-tuned ``AutoModelForSequenceClassification``
    (Legal-BERT or BERT) for legal clause categorisation.

    The model is loaded **lazily** in ``__init__`` and reused for all
    subsequent prediction calls — making this class suitable as a long-lived
    singleton inside a FastAPI application.

    Parameters
    ----------
    model_dir : str | Path | None
        Directory containing the saved model and tokenizer files.
        Defaults to ``config.MODEL_SAVE_DIR``.  Falls back to the HF Hub
        primary model name, then the fallback model name, so that predictions
        can still be made from a zero-shot / untrained model.
    device : str | None
        PyTorch device string.  ``None`` → auto-detect via :func:`utils.get_device`.
    confidence_threshold : float
        Predictions below this threshold are overridden with ``"other"``.

    Attributes
    ----------
    model : AutoModelForSequenceClassification | None
    tokenizer : AutoTokenizer | None
    device : str
    is_loaded : bool
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_dir: Path = Path(model_dir) if model_dir else MODEL_SAVE_DIR
        self.device: str = device or get_device()
        self.confidence_threshold: float = confidence_threshold

        self.model = None
        self.tokenizer = None
        self.is_loaded: bool = False

        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load the tokenizer and model from disk (or HF Hub as a fallback).

        Loading order:
        1. ``self.model_dir`` (fine-tuned local model)
        2. Primary HF Hub model (``nlpaueb/legal-bert-base-uncased``)
        3. Fallback HF Hub model (``bert-base-uncased``)
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # ---- Try loading fine-tuned model from disk --------------------
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            source = str(self.model_dir)
            logger.info("Loading fine-tuned model from %s", source)
        else:
            logger.warning(
                "Trained model not found at '%s'. "
                "Falling back to base model weights from Hugging Face Hub. "
                "Run train.py to fine-tune for best accuracy.",
                self.model_dir,
            )
            # Attempt primary Hub model, then fallback
            try:
                AutoTokenizer.from_pretrained(PRIMARY_MODEL_NAME)
                source = PRIMARY_MODEL_NAME
            except Exception:  # noqa: BLE001
                source = FALLBACK_MODEL_NAME

            logger.info("Loading base model: %s", source)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=NUM_LABELS,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("Classifier ready on device: %s", self.device)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load classifier model: %s", exc)
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, Any]:
        """
        Classify a single legal paragraph.

        Parameters
        ----------
        text : str
            Raw paragraph string.

        Returns
        -------
        dict
            ``{"clause_type": str, "confidence": float}``

        Example
        -------
        >>> clf = ClauseClassifier()
        >>> clf.predict("Either party may terminate with 30 days notice.")
        {'clause_type': 'termination', 'confidence': 0.9312}
        """
        if not self.is_loaded:
            logger.error("Model is not loaded – returning fallback result.")
            return format_prediction("other", 0.0)

        if not text or not text.strip():
            logger.warning("Received empty text – returning 'other'.")
            return format_prediction("other", 0.0)

        return self.predict_batch([text])[0]

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = INFERENCE_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """
        Classify a list of legal paragraph strings in mini-batches.

        Parameters
        ----------
        texts : list[str]
            Paragraphs to classify.
        batch_size : int
            Number of paragraphs per inference batch.  Smaller values use
            less GPU memory; larger values are faster.

        Returns
        -------
        list[dict]
            One ``{"clause_type": str, "confidence": float}`` dict per input.

        Example
        -------
        >>> clf = ClauseClassifier()
        >>> results = clf.predict_batch([
        ...     "The contract may be terminated by either party.",
        ...     "All payments are due within 30 days of invoice.",
        ... ])
        [
            {'clause_type': 'termination',  'confidence': 0.9102},
            {'clause_type': 'payment_terms','confidence': 0.8776},
        ]
        """
        if not self.is_loaded:
            logger.error("Model not loaded – returning fallback results.")
            return [format_prediction("other", 0.0) for _ in texts]

        import torch
        import torch.nn.functional as F

        all_results: list[dict[str, Any]] = []

        # Process in mini-batches to avoid OOM on large inputs
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            # Tokenise
            encoding = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Inference (no gradient tracking needed)
            with torch.no_grad():
                logits = self.model(**encoding).logits

            # Softmax → probabilities
            probabilities = F.softmax(logits, dim=-1)
            top_scores, top_indices = probabilities.max(dim=-1)

            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                label = ID2LABEL.get(idx, "other")

                # Apply confidence threshold
                if score < self.confidence_threshold:
                    logger.debug(
                        "Confidence %.4f < threshold %.4f → defaulting to 'other'",
                        score,
                        self.confidence_threshold,
                    )
                    label = "other"

                all_results.append(format_prediction(label, score))

        return all_results


# ===========================================================================
# Module-level convenience functions (FastAPI / pipeline entry points)
# ===========================================================================

def _get_classifier() -> ClauseClassifier:
    """Return (and lazily instantiate) the module-level classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = ClauseClassifier()
    return _classifier


def predict_clause(paragraph: str) -> dict[str, Any]:
    """
    Classify a single legal paragraph string.

    This is the **primary integration point** for the FastAPI backend and
    any downstream pipeline stage (risk detection, summarisation, etc.).

    Parameters
    ----------
    paragraph : str
        A single paragraph of legal contract text.

    Returns
    -------
    dict
        ``{"clause_type": str, "confidence": float}``

    Example
    -------
    ::

        from ai_engine.clause_classifier import predict_clause

        result = predict_clause(
            "The agreement may be terminated by either party with 30 days notice."
        )
        # → {'clause_type': 'termination', 'confidence': 0.91}
    """
    return _get_classifier().predict(paragraph)


def predict_batch_clauses(paragraphs: list[str]) -> list[dict[str, Any]]:
    """
    Classify multiple legal paragraph strings in a single call.

    Parameters
    ----------
    paragraphs : list[str]
        List of paragraph strings extracted from a contract.

    Returns
    -------
    list[dict]
        One result dict per input paragraph.

    Example
    -------
    ::

        from ai_engine.clause_classifier import predict_batch_clauses

        results = predict_batch_clauses([
            "Either party may terminate with 30 days notice.",
            "All payments must be made within 15 days of invoice.",
            "The laws of California shall govern this agreement.",
        ])
    """
    return _get_classifier().predict_batch(paragraphs)


# ===========================================================================
# Self-contained demo (run as __main__)
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    demo_paragraphs = [
        "The agreement may be terminated by either party with 30 days written notice.",
        "Client agrees to pay all invoices within fifteen (15) days of receipt.",
        "In no event shall either party be liable for indirect or consequential damages.",
        "Each party agrees to hold the confidential information of the other in strict confidence.",
        "This agreement shall be governed by the laws of the State of California.",
        "Each party shall indemnify and hold harmless the other from any third-party claims.",
        "All intellectual property created under this agreement remains the property of the client.",
        "Disputes shall be resolved through binding arbitration under the AAA Rules.",
        "Neither party shall be liable for delays caused by events beyond its reasonable control.",
        "This agreement constitutes the entire understanding between the parties.",
    ]

    print("\n" + "=" * 70)
    print("Clause Classifier – Batch Inference Demo")
    print("=" * 70)

    clf = ClauseClassifier()
    results = clf.predict_batch(demo_paragraphs)

    print(f"\n{'#':>3}  {'Confidence':>10}  {'Clause Type':<25}  {'Text'}")
    print("-" * 70)
    for i, (para, res) in enumerate(zip(demo_paragraphs, results), 1):
        print(
            f"{i:>3}  {res['confidence']:>10.4f}  "
            f"{res['clause_type']:<25}  {para[:50]}…"
        )

    print("\n" + "=" * 70)
    print("Single-call entry-point demo:")
    print("=" * 70)
    single = predict_clause("The governing law shall be the laws of the State of New York.")
    print(f"\npredict_clause() → {single}")
