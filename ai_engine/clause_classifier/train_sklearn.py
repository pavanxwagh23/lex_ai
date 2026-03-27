"""
train_sklearn.py – Fast lightweight clause classifier using TF-IDF + Logistic Regression.

No GPU or Hugging Face required. Trains in seconds and saves a .pkl model
that can be loaded instantly by predict_sklearn.py.

Usage
-----
From the project root:

    python ai_engine/clause_classifier/train_sklearn.py

Output
------
    models/sklearn_clause_classifier/model.pkl   ← pipeline (vectorizer + classifier)
    models/sklearn_clause_classifier/report.txt  ← classification report
"""

import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ai_engine.clause_classifier.config import CLAUSE_LABELS, DATA_PATH

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR  = _PROJECT_ROOT / "models" / "sklearn_clause_classifier"
MODEL_PATH = MODEL_DIR / "model.pkl"
REPORT_PATH = MODEL_DIR / "report.txt"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def train() -> None:
    log.info("=" * 60)
    log.info("Legal Clause Classifier – sklearn Training Pipeline")
    log.info("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df["text"]  = df["text"].str.strip()
    df["label"] = df["label"].str.strip()

    # Keep only recognised labels
    df = df[df["label"].isin(CLAUSE_LABELS)]
    log.info("Loaded %d rows  |  %d classes", len(df), df["label"].nunique())
    log.info("\nLabel distribution:\n%s", df["label"].value_counts().to_string())

    # ── 2. Train / validation split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    log.info("Train: %d  |  Validation: %d", len(X_train), len(X_test))

    # ── 3. Build pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=20_000,
            sublinear_tf=True,    # apply log normalization to term frequencies
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])

    # ── 4. Train ──────────────────────────────────────────────────────────────
    log.info("Training …")
    pipeline.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    log.info("\n%s", report)

    # ── 6. Save model and report ──────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Model saved  →  %s", MODEL_PATH)
    log.info("Report saved →  %s", REPORT_PATH)
    log.info("=" * 60)
    log.info("Training complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    train()
