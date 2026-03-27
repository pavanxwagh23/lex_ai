"""
config.py – Central configuration for the Clause Classifier module.

All hyperparameters, label mappings, and filesystem paths are defined here
so that every other module imports from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: ai_engine/clause_classifier/)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent          # …/ai_engine/clause_classifier
_AI_ENGINE_DIR = _MODULE_DIR.parent                    # …/ai_engine
PROJECT_ROOT = _AI_ENGINE_DIR.parent                   # …/legal_ai_analyzer

# ---------------------------------------------------------------------------
# Clause label taxonomy
# ---------------------------------------------------------------------------

CLAUSE_LABELS: list[str] = [
    "termination",
    "payment_terms",
    "liability",
    "confidentiality",
    "governing_law",
    "indemnification",
    "intellectual_property",
    "dispute_resolution",
    "force_majeure",
    "non_compete",
    "warranties",
    "other",
]

# Bidirectional mappings consumed by Hugging Face model config
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(CLAUSE_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(CLAUSE_LABELS)}

NUM_LABELS: int = len(CLAUSE_LABELS)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Primary model – domain-specific legal language model
PRIMARY_MODEL_NAME: str = "nlpaueb/legal-bert-base-uncased"

# Fallback model – standard BERT (used when primary cannot be downloaded)
FALLBACK_MODEL_NAME: str = "bert-base-uncased"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

MAX_LENGTH: int = 512        # Maximum tokenizer sequence length
BATCH_SIZE: int = 8          # Per-device train & eval batch size
EPOCHS: int = 3              # Number of fine-tuning epochs
LEARNING_RATE: float = 2e-5  # AdamW learning rate
WEIGHT_DECAY: float = 0.01   # Regularisation weight decay
WARMUP_RATIO: float = 0.1    # Fraction of steps for LR warmup
VALIDATION_SPLIT: float = 0.2  # 20 % of data reserved for validation
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

# Saved / fine-tuned model directory
MODEL_SAVE_DIR: Path = PROJECT_ROOT / "models" / "clause_classifier_model"

# Training dataset
DATA_PATH: Path = PROJECT_ROOT / "data" / "clause_dataset.csv"

# Logging directory for Hugging Face Trainer
LOGGING_DIR: Path = PROJECT_ROOT / "logs" / "clause_classifier"

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

# Minimum confidence required before returning a label (otherwise → "other")
MIN_CONFIDENCE_THRESHOLD: float = 0.30

# Number of paragraphs processed in a single inference batch
INFERENCE_BATCH_SIZE: int = 16
