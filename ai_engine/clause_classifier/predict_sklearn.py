"""
predict_sklearn.py – Fast inference using the trained sklearn TF-IDF + LR model.

Usage (module-level)
--------------------
    from ai_engine.clause_classifier.predict_sklearn import predict_clause_sklearn, predict_batch_sklearn

    result = predict_clause_sklearn("Either party may terminate with 30 days notice.")
    # → {"clause_type": "termination", "confidence": 0.96}

Usage (FastAPI integration)
---------------------------
    from ai_engine.clause_classifier.predict_sklearn import predict_clause_sklearn

    @app.post("/classify")
    async def classify(text: str):
        return predict_clause_sklearn(text)
"""

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
MODEL_PATH    = _PROJECT_ROOT / "models" / "sklearn_clause_classifier" / "model.pkl"

# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline = None


def sklearn_model_available() -> bool:
    """Return True if the trained sklearn pipeline file exists on disk."""
    return MODEL_PATH.is_file()


def _load_model():
    """Load and cache the sklearn pipeline from disk."""
    global _pipeline
    if _pipeline is None:
        try:
            import joblib
            _pipeline = joblib.load(MODEL_PATH)
            log.info("sklearn clause classifier loaded from %s", MODEL_PATH)
        except FileNotFoundError:
            log.error(
                "Model not found at %s. Run train_sklearn.py first:\n"
                "  python ai_engine/clause_classifier/train_sklearn.py",
                MODEL_PATH,
            )
            _pipeline = None
    return _pipeline


def predict_clause_sklearn(text: str) -> dict[str, Any]:
    """
    Classify a single legal clause paragraph.

    Parameters
    ----------
    text : str
        A single legal clause or contract sentence.

    Returns
    -------
    dict
        {"clause_type": str, "confidence": float}
        Falls back to {"clause_type": "other", "confidence": 0.0} if model not loaded.
    """
    if not text or not text.strip():
        return {"clause_type": "other", "confidence": 0.0}

    pipeline = _load_model()
    if pipeline is None:
        return {"clause_type": "other", "confidence": 0.0}

    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    confidence = float(max(proba))
    return {"clause_type": label, "confidence": round(confidence, 4)}


def predict_batch_sklearn(texts: list[str]) -> list[dict[str, Any]]:
    """
    Classify a list of legal clause paragraphs in one call.

    Parameters
    ----------
    texts : list[str]
        Paragraphs to classify.

    Returns
    -------
    list[dict]
        One {"clause_type": str, "confidence": float} per input.
    """
    pipeline = _load_model()
    if pipeline is None:
        return [{"clause_type": "other", "confidence": 0.0} for _ in texts]

    labels      = pipeline.predict(texts)
    proba_matrix = pipeline.predict_proba(texts)

    return [
        {"clause_type": lbl, "confidence": round(float(max(proba)), 4)}
        for lbl, proba in zip(labels, proba_matrix)
    ]


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    test_clauses = [
        "Either party may terminate this agreement with 30 days written notice.",
        "Client agrees to pay all invoices within fifteen (15) days of the invoice date.",
        "In no event shall either party be liable for indirect or consequential damages.",
        "Each party agrees to hold the confidential information of the other in strict confidence.",
        "This agreement shall be governed by the laws of the State of California.",
        "Each party shall indemnify and hold harmless the other from any third-party claims.",
        "All intellectual property created under this agreement shall remain the property of the client.",
        "Any disputes arising from this agreement shall be resolved by binding arbitration.",
        "Neither party shall be liable for delays caused by events beyond its reasonable control.",
        "The employee shall not work for a competitor for 12 months after termination.",
        "The vendor represents and warrants that its software does not infringe any third-party IP.",
        "This agreement constitutes the entire understanding between the parties.",
    ]

    print("\n" + "=" * 75)
    print("sklearn Clause Classifier – Inference Demo")
    print("=" * 75)
    results = predict_batch_sklearn(test_clauses)
    print(f"\n{'#':>3}  {'Confidence':>10}  {'Clause Type':<25}  Text")
    print("-" * 75)
    for i, (clause, res) in enumerate(zip(test_clauses, results), 1):
        print(f"{i:>3}  {res['confidence']:>10.4f}  {res['clause_type']:<25}  {clause[:45]}…")
    print("=" * 75)
