"""
train.py – Full fine-tuning pipeline for the Legal Clause Classifier.

Usage
-----
From the project root run::

    python ai_engine/clause_classifier/train.py

The script will:
1. Resolve the best available model (legal-bert → bert fallback).
2. Load and validate ``data/clause_dataset.csv``.
3. Tokenize and split the data 80 / 20 train / validation.
4. Fine-tune using the Hugging Face ``Trainer`` API.
5. Evaluate and print a full classification report.
6. Save the trained model + tokenizer to ``models/clause_classifier_model/``.

Requirements
------------
    pip install transformers datasets torch scikit-learn pandas accelerate
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (for running as a script)
# ---------------------------------------------------------------------------
_CLAUSE_CLASSIFIER_DIR = Path(__file__).resolve().parent
_AI_ENGINE_DIR = _CLAUSE_CLASSIFIER_DIR.parent
_PROJECT_ROOT = _AI_ENGINE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ai_engine.clause_classifier.config import (
    BATCH_SIZE,
    CLAUSE_LABELS,
    DATA_PATH,
    EPOCHS,
    FALLBACK_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    LOGGING_DIR,
    MAX_LENGTH,
    MODEL_SAVE_DIR,
    NUM_LABELS,
    PRIMARY_MODEL_NAME,
    RANDOM_SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from ai_engine.clause_classifier.dataset_loader import (
    encode_labels,
    load_csv_dataset,
    tokenize_dataset,
)
from ai_engine.clause_classifier.utils import (
    compute_metrics,
    get_device,
    log_classification_report,
    resolve_model_name,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Execute the full training pipeline end-to-end.

    Raises
    ------
    SystemExit
        If the dataset file is missing or contains no valid rows.
    """
    logger.info("=" * 60)
    logger.info("Legal Clause Classifier – Fine-Tuning Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 – Resolve model name (legal-bert or bert fallback)
    # ------------------------------------------------------------------
    logger.info("Step 1/7: Resolving base model …")
    model_name = resolve_model_name(PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME)
    logger.info("Using model: %s", model_name)

    # ------------------------------------------------------------------
    # Step 2 – Load tokenizer
    # ------------------------------------------------------------------
    logger.info("Step 2/7: Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Step 3 – Load and prepare dataset
    # ------------------------------------------------------------------
    logger.info("Step 3/7: Loading dataset from %s …", DATA_PATH)
    try:
        df = load_csv_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    df = encode_labels(df, LABEL2ID)

    # ------------------------------------------------------------------
    # Step 4 – Tokenize and split
    # ------------------------------------------------------------------
    logger.info("Step 4/7: Tokenising and splitting dataset …")
    dataset_dict = tokenize_dataset(
        df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        seed=RANDOM_SEED,
    )
    train_dataset = dataset_dict["train"]
    val_dataset   = dataset_dict["validation"]

    # ------------------------------------------------------------------
    # Step 5 – Instantiate model
    # ------------------------------------------------------------------
    logger.info("Step 5/7: Loading base model for sequence classification …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # classification head is randomly initialised
    )

    device = get_device()
    logger.info("Training on device: %s", device)

    # ------------------------------------------------------------------
    # Step 6 – Configure Trainer
    # ------------------------------------------------------------------
    logger.info("Step 6/7: Configuring Trainer …")

    # Create output directories
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        # ---- Output ----
        output_dir=str(MODEL_SAVE_DIR),
        logging_dir=str(LOGGING_DIR),
        # ---- Training schedule ----
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        # ---- Evaluation & checkpointing ----
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # ---- Reproducibility ----
        seed=RANDOM_SEED,
        # ---- Logging ----
        logging_steps=10,
        report_to="none",   # Disable WandB / TensorBoard integrations by default
        # ---- Misc ----
        disable_tqdm=False,
        dataloader_num_workers=0,  # Keep 0 for compatibility on all platforms
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ------------------------------------------------------------------
    # Step 7 – Train, evaluate, and save
    # ------------------------------------------------------------------
    logger.info("Step 7/7: Starting training …")
    trainer.train()

    logger.info("Evaluating on validation set …")
    eval_results = trainer.evaluate()
    logger.info("Validation results: %s", eval_results)

    # Detailed per-class classification report
    predictions_output = trainer.predict(val_dataset)
    import numpy as np
    y_pred = np.argmax(predictions_output.predictions, axis=-1).tolist()
    y_true = predictions_output.label_ids.tolist()
    log_classification_report(y_true, y_pred, CLAUSE_LABELS)

    # Save model + tokenizer
    logger.info("Saving fine-tuned model to %s …", MODEL_SAVE_DIR)
    trainer.save_model(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))

    logger.info("=" * 60)
    logger.info("Training complete!  Model saved at: %s", MODEL_SAVE_DIR)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
