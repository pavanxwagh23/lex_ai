"""Tests for clause mapping (splitting + rule/sklearn classification)."""

from backend.services.clause_service import map_clauses


def test_map_clauses_handles_single_newline_extraction():
    """PDF-style text often uses single newlines, not blank-line paragraphs."""
    text = (
        "Either party may terminate this agreement with thirty days written notice.\n"
        "Client shall pay all invoices within fifteen days of the invoice date.\n"
        "Each party agrees to keep confidential information strictly confidential.\n"
    )
    result = map_clauses(text)
    assert result.get("error") is None
    assert result["total_paragraphs"] >= 2
    assert result["unique_clause_types"] >= 1
    assert result["classifier"] in ("sklearn", "rules")
    assert result["clause_groups"]


def test_map_clauses_chunks_very_long_single_line():
    sentence = (
        "The vendor shall indemnify the client against third-party claims arising from the services. "
    )
    text = (sentence * 30).strip()
    assert "\n\n" not in text
    result = map_clauses(text)
    assert result.get("error") is None
    assert result["total_paragraphs"] >= 1
    assert result["all_results"]
