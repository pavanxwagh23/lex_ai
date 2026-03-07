"""
risk_types.py
=============
Core type definitions for the Risk Detection Engine.

Provides enumerations for risk categories and severity levels, as well as
immutable dataclasses that represent a single detected risk and the full
analysis result returned by :class:`~risk_engine.RiskDetectionEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskCategory(str, Enum):
    """All supported legal-risk categories.

    Using ``str`` as a mixin makes each member JSON-serialisable without a
    custom encoder.
    """

    UNLIMITED_LIABILITY = "Unlimited Liability"
    MISSING_TERMINATION = "Missing Termination Clause"
    ONE_SIDED_INDEMNITY = "One-sided Indemnity"
    PAYMENT_RISK = "Payment Risk"
    JURISDICTION_RISK = "Jurisdiction Risk"
    CONFIDENTIALITY_RISK = "Confidentiality Risk"
    IP_OWNERSHIP_RISK = "Intellectual Property Ownership Risk"
    PENALTY_CLAUSE = "Penalty Clauses"


class RiskSeverity(str, Enum):
    """Severity levels for a detected risk."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def score(self) -> int:
        """Numerical weight used in the overall risk-score calculation."""
        _weights: dict[RiskSeverity, int] = {
            RiskSeverity.LOW: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.HIGH: 3,
        }
        return _weights[self]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskMatch:
    """Represents a single risk finding within a paragraph.

    Attributes
    ----------
    category:
        The :class:`RiskCategory` that was triggered.
    severity:
        The :class:`RiskSeverity` assigned to this finding.
    text:
        The original paragraph (or sentence) that triggered the rule.
    explanation:
        Human-readable description of *why* this passage is considered risky.
    matched_phrase:
        The specific keyword / phrase inside *text* that matched the rule.
    paragraph_index:
        Zero-based index of the paragraph in the input list.
    """

    category: RiskCategory
    severity: RiskSeverity
    text: str
    explanation: str
    matched_phrase: str
    paragraph_index: int = 0


@dataclass
class RiskAnalysisResult:
    """Aggregated result returned by :meth:`~risk_engine.RiskDetectionEngine.analyze`.

    Attributes
    ----------
    risk_score:
        Overall risk score normalised to the range **0 – 10**.
    risks_detected:
        Ordered list of :class:`RiskMatch` objects (highest severity first).
    total_paragraphs:
        Number of input paragraphs that were analysed.
    risky_paragraph_count:
        Number of paragraphs that contained at least one risk.
    """

    risk_score: float
    risks_detected: List[RiskMatch]
    total_paragraphs: int
    risky_paragraph_count: int

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain ``dict`` suitable for JSON serialisation."""
        return {
            "risk_score": round(self.risk_score, 2),
            "total_paragraphs": self.total_paragraphs,
            "risky_paragraph_count": self.risky_paragraph_count,
            "risks_detected": [
                {
                    "type": match.category.value,
                    "severity": match.severity.value,
                    "text": match.text,
                    "explanation": match.explanation,
                    "matched_phrase": match.matched_phrase,
                    "paragraph_index": match.paragraph_index,
                }
                for match in self.risks_detected
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RiskAnalysisResult("
            f"risk_score={self.risk_score:.2f}, "
            f"risks={len(self.risks_detected)}, "
            f"paragraphs={self.total_paragraphs})"
        )


# ---------------------------------------------------------------------------
# Sentinel for "no risk found"
# ---------------------------------------------------------------------------

EMPTY_RESULT = RiskAnalysisResult(
    risk_score=0.0,
    risks_detected=[],
    total_paragraphs=0,
    risky_paragraph_count=0,
)
