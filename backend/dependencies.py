"""
backend/dependencies.py
========================
Shared FastAPI dependency functions injected into route handlers.

Using FastAPI's ``Depends`` system keeps routes thin and testable —
all cross-cutting concerns (registry look-up, file validation, etc.)
live here rather than being duplicated across route files.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from backend.services.document_service import get_contract
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def get_contract_or_404(contract_id: str) -> dict:
    """
    FastAPI dependency: look up a contract in the registry or raise HTTP 404.

    Usage in a route::

        @router.get("/{contract_id}")
        async def my_route(contract: dict = Depends(get_contract_or_404)):
            ...

    Parameters
    ----------
    contract_id : str
        Path parameter extracted from the URL.

    Returns
    -------
    dict
        The full registry record for the contract.

    Raises
    ------
    HTTPException(404)
        If no contract with the given ID exists in the in-memory registry.
    """
    try:
        return get_contract(contract_id)
    except KeyError:
        logger.warning("Contract not found: %s", contract_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract '{contract_id}' not found. "
                   "Please upload the contract first via POST /contracts/upload.",
        )
