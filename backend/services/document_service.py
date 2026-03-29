"""
backend/services/document_service.py
======================================
Business logic for contract file management and the in-memory registry.

The in-memory registry (``contracts_registry``) simulates a database until a
real persistence layer is added.  It is an application-scoped dictionary
keyed by ``contract_id`` (UUID hex string).

Registry entry shape
--------------------
::

    contracts_registry["<uuid-hex>"] = {
        "contract_id":    "<uuid-hex>",
        "filename":       "agreement.pdf",
        "file_path":      "/absolute/path/to/storage/contracts/<uuid>_agreement.pdf",
        "uploaded_at":    "2026-03-07T17:15:00Z",
        "file_size_bytes": 102400,
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import ALLOWED_EXTENSIONS
from backend.schemas.contract_schema import ContractMetadata, UploadContractResponse
from backend.utils.file_storage import (
    generate_contract_id,
    save_uploaded_file,
    validate_extension,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory contract registry
# ---------------------------------------------------------------------------
#: Application-lifetime registry.  Keyed by contract_id (UUID hex string).
contracts_registry: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def register_contract(
    filename: str,
    content:  bytes,
) -> UploadContractResponse:
    """
    Validate, store, and register an uploaded contract file.

    Parameters
    ----------
    filename : str
        Original filename submitted by the client.
    content : bytes
        Raw file bytes from the upload stream.

    Returns
    -------
    UploadContractResponse
        Structured metadata for the newly registered contract.

    Raises
    ------
    ValueError
        If the file extension is not in :data:`~backend.config.ALLOWED_EXTENSIONS`.
    """
    if not validate_extension(filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"File type not allowed: '{Path(filename).suffix}'. "
            f"Accepted types: {allowed}"
        )

    contract_id  = generate_contract_id()
    uploaded_at  = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    file_path    = save_uploaded_file(contract_id, filename, content)

    record: dict[str, Any] = {
        "contract_id":     contract_id,
        "filename":        filename,
        "file_path":       str(file_path),
        "uploaded_at":     uploaded_at,
        "file_size_bytes": len(content),
    }
    contracts_registry[contract_id] = record

    logger.info(
        "Registered contract id=%s  filename=%s  size=%d B",
        contract_id,
        filename,
        len(content),
    )

    return UploadContractResponse(
        contract_id = contract_id,
        filename    = filename,
        file_path   = str(file_path),
        status      = "uploaded",
        uploaded_at = uploaded_at,
    )


def get_contract(contract_id: str) -> dict[str, Any]:
    """
    Retrieve contract metadata from the in-memory registry.

    Parameters
    ----------
    contract_id : str
        UUID hex string.

    Returns
    -------
    dict
        Full registry record for the contract.

    Raises
    ------
    KeyError
        If no contract with the given ID exists in the registry.
    """
    record = contracts_registry.get(contract_id)
    if record is None:
        raise KeyError(f"Contract '{contract_id}' not found in registry.")
    return record


def list_contracts() -> list[ContractMetadata]:
    """
    Return a list of all contracts currently in the in-memory registry.

    Returns
    -------
    list[ContractMetadata]
        Sorted by upload time (most recent first).
    """
    return [
        ContractMetadata(**record)
        for record in sorted(
            contracts_registry.values(),
            key=lambda r: r["uploaded_at"],
            reverse=True,
        )
    ]


def get_file_text(contract_id: str) -> tuple[str, Path]:
    """
    Retrieve the registry record and verify the file exists on disk.

    A convenience helper used by ``analysis_service`` and
    ``comparison_service`` before calling the AI pipeline.

    Parameters
    ----------
    contract_id : str
        UUID hex string.

    Returns
    -------
    tuple[str, Path]
        ``(file_path_str, Path_object)`` for the stored contract file.

    Raises
    ------
    KeyError
        If the contract ID is not in the registry.
    FileNotFoundError
        If the file has been deleted from disk since registration.
    """
    record = get_contract(contract_id)
    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(
            f"File for contract '{contract_id}' not found on disk: {file_path}"
        )
    return str(file_path), file_path

from backend.ai_client.ai_pipeline import extract_text

def get_contract_text(contract_id: str) -> str:
    """
    Fetch contract text from storage.
    For now, simulate with local file read or mock.
    """
    try:
        file_path_str, _ = get_file_text(contract_id)
        return extract_text(file_path_str)
    except Exception as e:
        logger.warning("Failed to extract text for %s: %s", contract_id, e)
        # Mock fallback for development if file not found
        return "This is a mock contract text for " + contract_id

