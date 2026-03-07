"""
backend/services/comparison_service.py
=========================================
Business logic for semantic comparison of two uploaded contracts.
"""

from __future__ import annotations

from backend.ai_client import ai_pipeline
from backend.schemas.analysis_schema import (
    ClauseMatchItem,
    CompareContractsResponse,
)
from backend.services.document_service import get_file_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def run_comparison(contract_id_a: str, contract_id_b: str) -> CompareContractsResponse:
    """
    Semantically compare two uploaded contracts and return a structured diff.

    Parameters
    ----------
    contract_id_a : str
        UUID of the **original** contract (Document A).
    contract_id_b : str
        UUID of the **revised** contract (Document B).

    Returns
    -------
    CompareContractsResponse
        Structured diff with ``added_clauses``, ``removed_clauses``,
        ``modified_clauses``, and ``similar_clauses``.

    Raises
    ------
    KeyError
        If either contract ID is not in the in-memory registry.
    FileNotFoundError
        If either contract file is missing from disk.
    """
    logger.info(
        "Starting comparison: contract_a=%s  contract_b=%s",
        contract_id_a, contract_id_b,
    )

    # --- locate and read both documents ---
    path_a, _ = get_file_text(contract_id_a)
    path_b, _ = get_file_text(contract_id_b)

    text_a = ai_pipeline.extract_text(path_a)
    text_b = ai_pipeline.extract_text(path_b)

    # --- delegate to AI pipeline ---
    result = ai_pipeline.compare_contracts(text_a, text_b)

    # --- map raw dicts → typed ClauseMatchItem models ---
    def _to_match_items(raw_list: list) -> list[ClauseMatchItem]:
        items = []
        for r in raw_list:
            if isinstance(r, dict):
                items.append(
                    ClauseMatchItem(
                        clause_a   = r.get("clause_a", ""),
                        clause_b   = r.get("clause_b"),
                        similarity = float(r.get("similarity", -1.0)),
                        status     = r.get("status", "unknown"),
                    )
                )
        return items

    response = CompareContractsResponse(
        contract_a       = contract_id_a,
        contract_b       = contract_id_b,
        added_clauses    = result.get("added_clauses", []),
        removed_clauses  = _to_match_items(result.get("removed_clauses", [])),
        modified_clauses = _to_match_items(result.get("modified_clauses", [])),
        similar_clauses  = _to_match_items(result.get("similar_clauses", [])),
        metadata         = result.get("metadata", {}),
    )

    logger.info(
        "Comparison complete: added=%d  removed=%d  modified=%d  similar=%d",
        len(response.added_clauses),
        len(response.removed_clauses),
        len(response.modified_clauses),
        len(response.similar_clauses),
    )

    return response
