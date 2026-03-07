"""
backend/schemas/analysis_schema.py
=====================================
Pydantic models for analysis, summary, and comparison API responses.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Analysis response
# ---------------------------------------------------------------------------

class RiskItem(BaseModel):
    """A single detected risk finding."""
    category:    str   = Field(..., description="Risk category, e.g. 'liability'")
    description: str   = Field(..., description="Human-readable risk description")
    severity:    str   = Field(..., description="'low', 'medium', or 'high'")
    paragraph:   Optional[str] = Field(None, description="Source paragraph text")


class AnalysisResponse(BaseModel):
    """
    Full analysis result returned by ``POST /contracts/{contract_id}/analyze``.
    """

    contract_id: str              = Field(..., description="UUID contract identifier")
    risk_score:  float            = Field(..., description="Normalised risk score 0–10")
    clauses:     dict[str, str]   = Field(
        ...,
        description="Detected clause types mapped to their representative text",
    )
    risks:       list[RiskItem]   = Field(
        ...,
        description="List of detected legal risk findings",
    )
    risk_summary: list[str]       = Field(
        default_factory=list,
        description="Short human-readable risk messages (for display cards)",
    )
    metadata:    dict[str, Any]   = Field(
        default_factory=dict,
        description="Pipeline run-time statistics",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "contract_id": "8b31f9a2c14b4e0d9a2b3c4d5e6f7a8b",
            "risk_score":  7.2,
            "clauses": {
                "termination":  "Either party may terminate with 30 days notice.",
                "payment_terms":"Client must pay within 30 days of invoice.",
                "liability":    "Neither party is liable for indirect damages.",
            },
            "risks": [
                {
                    "category":    "liability",
                    "description": "Unlimited liability exposure detected",
                    "severity":    "high",
                    "paragraph":   "In no event shall either party …",
                }
            ],
            "risk_summary": ["Unlimited liability detected"],
            "metadata": {"paragraphs_analysed": 12, "engine": "RiskDetectionEngine"},
        }
    }}


# ---------------------------------------------------------------------------
# Summary response
# ---------------------------------------------------------------------------

class SummaryResponse(BaseModel):
    """
    Returned by ``POST /contracts/{contract_id}/summary``.
    """

    contract_id: str        = Field(..., description="UUID contract identifier")
    summary:     list[str]  = Field(
        ...,
        description="Bullet-point list of key contract terms",
    )
    full_summary: Optional[str] = Field(
        None,
        description="Prose summary of the entire document",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "contract_id": "8b31f9a2c14b4e0d9a2b3c4d5e6f7a8b",
            "summary": [
                "Contract duration is 2 years.",
                "Vendor must provide monthly services.",
                "Termination requires 30-day written notice.",
            ],
            "full_summary": "This agreement between Company A and Company B …",
        }
    }}


# ---------------------------------------------------------------------------
# Compare request + response
# ---------------------------------------------------------------------------

class CompareContractsRequest(BaseModel):
    """
    Request body for ``POST /contracts/compare``.
    """

    contract_a: str = Field(..., description="UUID of the original contract")
    contract_b: str = Field(..., description="UUID of the revised contract")

    model_config = {"json_schema_extra": {
        "example": {
            "contract_a": "uuid-of-original",
            "contract_b": "uuid-of-revised",
        }
    }}


class ClauseMatchItem(BaseModel):
    """A single matched clause pair between two contracts."""
    clause_a:   str           = Field(..., description="Clause text from Contract A")
    clause_b:   Optional[str] = Field(None, description="Matching clause text from Contract B")
    similarity: float         = Field(..., description="Cosine similarity score [0,1]")
    status:     str           = Field(..., description="identical | modified | related | removed")


class CompareContractsResponse(BaseModel):
    """
    Returned by ``POST /contracts/compare``.
    """

    contract_a:      str                  = Field(..., description="UUID of Contract A")
    contract_b:      str                  = Field(..., description="UUID of Contract B")
    added_clauses:   list[str]            = Field(default_factory=list)
    removed_clauses: list[ClauseMatchItem] = Field(default_factory=list)
    modified_clauses:list[ClauseMatchItem] = Field(default_factory=list)
    similar_clauses: list[ClauseMatchItem] = Field(default_factory=list)
    metadata:        dict[str, Any]        = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {
            "contract_a": "uuid-a",
            "contract_b": "uuid-b",
            "added_clauses":  ["The vendor must comply with GDPR."],
            "removed_clauses": [],
            "modified_clauses": [
                {
                    "clause_a": "Termination with 30 days notice.",
                    "clause_b": "Termination with 60 days notice.",
                    "similarity": 0.82,
                    "status": "modified",
                }
            ],
            "similar_clauses": [],
            "metadata": {"model": "all-MiniLM-L6-v2"},
        }
    }}
