"""
backend/schemas/contract_schema.py
====================================
Pydantic request and response models for contract upload and registry operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadContractResponse(BaseModel):
    """
    Returned after a successful contract upload.

    Attributes
    ----------
    contract_id : str
        UUID hex identifier assigned to this contract.
    filename : str
        Original filename submitted by the client.
    file_path : str
        Absolute path where the file was stored on disk.
    status : str
        Always ``"uploaded"`` on success.
    uploaded_at : str
        ISO-8601 UTC timestamp of the upload.
    """

    contract_id: str  = Field(..., description="UUID contract identifier")
    filename:    str  = Field(..., description="Original filename")
    file_path:   str  = Field(..., description="Server-side storage path")
    status:      str  = Field(default="uploaded")
    uploaded_at: str  = Field(..., description="ISO-8601 UTC upload timestamp")

    model_config = {"json_schema_extra": {
        "example": {
            "contract_id": "8b31f9a2c14b4e0d9a2b3c4d5e6f7a8b",
            "filename":    "agreement.pdf",
            "file_path":   "/storage/contracts/8b31f9a2_agreement.pdf",
            "status":      "uploaded",
            "uploaded_at": "2026-03-07T17:15:00Z",
        }
    }}


class ContractMetadata(BaseModel):
    """
    In-memory registry record for a single uploaded contract.

    This mirrors the structure of the ``contracts_registry`` dictionary entries.
    """

    contract_id: str
    filename:    str
    file_path:   str
    uploaded_at: str
    file_size_bytes: Optional[int] = None


class ContractListResponse(BaseModel):
    """Response for listing all contracts currently in the in-memory registry."""

    contracts: list[ContractMetadata]
    total:     int
