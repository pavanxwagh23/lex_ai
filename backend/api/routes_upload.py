"""
backend/api/routes_upload.py
==============================
FastAPI router for contract upload endpoints.

Routes
------
POST /contracts/upload   — Upload a PDF or DOCX contract file.
GET  /contracts          — List all contracts in the in-memory registry.
GET  /contracts/{id}     — Get metadata for a single contract.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from backend.dependencies import get_contract_or_404
from backend.schemas.contract_schema import (
    ContractListResponse,
    ContractMetadata,
    UploadContractResponse,
)
from backend.services.document_service import list_contracts, register_contract
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/contracts", tags=["Contracts"])


# ---------------------------------------------------------------------------
# POST /contracts/upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadContractResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a contract",
    description=(
        "Upload a PDF or DOCX legal contract. "
        "The file is saved locally, assigned a UUID, and registered "
        "in the in-memory contract registry."
    ),
)
async def upload_contract(
    file: UploadFile = File(..., description="PDF or DOCX contract file"),
) -> UploadContractResponse:
    """
    Accept a contract file upload and register it for AI processing.

    **Accepted types:** ``.pdf``, ``.docx``

    **Max size:** 50 MB (configurable via ``MAX_UPLOAD_BYTES`` env var)

    Returns a ``contract_id`` UUID that must be used in subsequent
    ``/analyze``, ``/summary``, and ``/compare`` requests.
    """
    # --- validate filename / extension ---
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    from pathlib import Path
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- read content and check size ---
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {mb} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info("Upload request: filename=%s  size=%d B", file.filename, len(content))

    # --- delegate to service layer ---
    try:
        response = register_contract(
            filename=file.filename,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except OSError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the uploaded file. Please try again.",
        )

    return response


# ---------------------------------------------------------------------------
# GET /contracts
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=ContractListResponse,
    summary="List all uploaded contracts",
)
async def list_all_contracts() -> ContractListResponse:
    """Return all contracts currently in the in-memory registry."""
    contracts = list_contracts()
    return ContractListResponse(contracts=contracts, total=len(contracts))


# ---------------------------------------------------------------------------
# GET /contracts/{contract_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{contract_id}",
    response_model=ContractMetadata,
    summary="Get contract metadata",
)
async def get_contract_metadata(
    contract: dict = Depends(get_contract_or_404),
) -> ContractMetadata:
    """Retrieve metadata for a single uploaded contract by its UUID."""
    return ContractMetadata(**contract)
