"""
dataset_loader.py – Data loading and preprocessing for Clause Classifier training.

Pipeline
--------
1. ``load_csv_dataset``   – Read CSV, validate schema, drop bad rows.
2. ``encode_labels``      – Map string labels → integer IDs.
3. ``tokenize_dataset``   – Tokenize text and return a HF DatasetDict.

The returned ``DatasetDict`` is split 80 / 20 train / validation with
stratification so that each class is proportionally represented in both
splits even with small datasets.
"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from .config import CLAUSE_LABELS, LABEL2ID, RANDOM_SEED, VALIDATION_SPLIT

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Expected CSV column names
_TEXT_COL  = "text"
_LABEL_COL = "label"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate the clause dataset CSV file.

    The CSV must contain at minimum two columns:
    - ``text``  : the raw paragraph string
    - ``label`` : the clause category string (must be in ``CLAUSE_LABELS``)

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the ``clause_dataset.csv`` file.

    Returns
    -------
    pd.DataFrame
        Validated dataframe with ``text`` and ``label`` columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to a readable file.
    ValueError
        If required columns are missing or no valid rows remain after cleaning.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    # ----- Column validation ------------------------------------------------
    missing_cols = {_TEXT_COL, _LABEL_COL} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # ----- Strip whitespace -------------------------------------------------
    df[_TEXT_COL]  = df[_TEXT_COL].astype(str).str.strip()
    df[_LABEL_COL] = df[_LABEL_COL].astype(str).str.strip()

    # ----- Drop nulls / empty rows ------------------------------------------
    before = len(df)
    df = df.dropna(subset=[_TEXT_COL, _LABEL_COL])
    df = df[df[_TEXT_COL].str.len() > 0]
    df = df[df[_LABEL_COL].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with empty text or label.", dropped)

    # ----- Label validation -------------------------------------------------
    invalid_labels = set(df[_LABEL_COL].unique()) - set(CLAUSE_LABELS)
    if invalid_labels:
        logger.warning(
            "Dropping %d rows with unrecognised labels: %s",
            len(df[df[_LABEL_COL].isin(invalid_labels)]),
            invalid_labels,
        )
        df = df[~df[_LABEL_COL].isin(invalid_labels)]

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset CSV.")

    logger.info(
        "After cleaning: %d rows | label distribution:\n%s",
        len(df),
        df[_LABEL_COL].value_counts().to_string(),
    )
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label2id: dict[str, int] | None = None) -> pd.DataFrame:
    """
    Add an integer ``label_id`` column to the dataframe by mapping string
    labels through the provided ``label2id`` dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe returned by :func:`load_csv_dataset`.
    label2id : dict[str, int] | None
        Label-to-index mapping. Defaults to :data:`config.LABEL2ID`.

    Returns
    -------
    pd.DataFrame
        Original dataframe with a new ``label_id`` column (``int``).
    """
    if label2id is None:
        label2id = LABEL2ID

    df = df.copy()
    df["label_id"] = df[_LABEL_COL].map(label2id)

    # Safety-check: any labels that slipped through without an ID?
    unmapped = df[df["label_id"].isna()]
    if not unmapped.empty:
        logger.error("Unmapped labels found:\n%s", unmapped[_LABEL_COL].value_counts())
        raise ValueError("Some labels could not be mapped to integer IDs.")

    df["label_id"] = df["label_id"].astype(int)
    return df


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """
    Split dataframe into train / validation sets and tokenize both.

    Steps
    -----
    1. Stratified train/val split (preserves class proportions).
    2. Convert each split to a Hugging Face ``Dataset``.
    3. Apply the tokenizer with padding and truncation.
    4. Rename ``label_id`` → ``labels`` (required by HF Trainer).
    5. Set dataset format to PyTorch tensors.

    Parameters
    ----------
    df : pd.DataFrame
        Labelled dataframe (must have ``text`` and ``label_id`` columns).
    tokenizer : PreTrainedTokenizerBase
        Hugging Face tokenizer matching the model to be fine-tuned.
    max_length : int
        Maximum token sequence length (sequences are padded / truncated).
    validation_split : float
        Fraction of data to use for validation (default: 0.20).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DatasetDict
        Keys ``"train"`` and ``"validation"``, each a ``Dataset`` with
        fields: ``input_ids``, ``attention_mask``, ``token_type_ids``
        (if BERT-family), and ``labels``.
    """
    # ----- Stratified split ------------------------------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        stratify=df["label_id"],
        random_state=seed,
    )
    logger.info("Train: %d rows | Validation: %d rows", len(train_df), len(val_df))

    # ----- HF Dataset features schema --------------------------------------
    features = Features(
        {
            _TEXT_COL: Value("string"),
            "label_id": ClassLabel(names=CLAUSE_LABELS),
        }
    )

    train_dataset = Dataset.from_pandas(
        train_df[[_TEXT_COL, "label_id"]].reset_index(drop=True),
        features=features,
    )
    val_dataset = Dataset.from_pandas(
        val_df[[_TEXT_COL, "label_id"]].reset_index(drop=True),
        features=features,
    )

    # ----- Tokenize ---------------------------------------------------------
    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[_TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(_tokenize, batched=True, desc="Tokenising train")
    val_dataset   = val_dataset.map(_tokenize, batched=True, desc="Tokenising val")

    # ----- Rename label column → "labels" (HF Trainer expects this name) ---
    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset   = val_dataset.rename_column("label_id", "labels")

    # ----- Set PyTorch tensor format ----------------------------------------
    columns = ["input_ids", "attention_mask", "labels"]
    # Some BERT models also produce token_type_ids
    if "token_type_ids" in train_dataset.column_names:
        columns.append("token_type_ids")

    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})
