"""
risk_rules.py
=============
Configurable rule definitions for the Risk Detection Engine.

Each :class:`RiskRule` bundles together everything the engine needs to
detect and explain a single risk category:

* Which keywords / phrases to look for
* The default severity
* A template explanation generator

Rules are registered in ``RISK_RULES`` — a plain list that the engine
iterates over.  To add a new risk category, simply append a new
:class:`RiskRule` instance to that list (no engine code changes needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .risk_types import RiskCategory, RiskSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RiskRule definition
# ---------------------------------------------------------------------------


@dataclass
class RiskRule:
    """Configuration for a single detectable legal risk.

    Attributes
    ----------
    category:
        The :class:`~risk_types.RiskCategory` this rule covers.
    severity:
        Default severity level assigned when this rule fires.
    keywords:
        List of phrases that indicate the presence of this risk.
        Matching is case-insensitive (handled in :mod:`risk_utils`).
    explanation_fn:
        Callable ``(paragraph: str, matched_phrase: str) -> str`` that
        produces a human-readable explanation.  Defaults to a generic
        template if not provided.
    whole_word:
        If ``True``, keywords must appear as whole words to avoid partial
        matches (e.g. "penalty" in "penalty clause" vs "penaltybox").
        Defaults to ``False`` for legal text (phrases are usually clear).
    enabled:
        Toggle a rule on/off without removing it from the list.
    """

    category: RiskCategory
    severity: RiskSeverity
    keywords: List[str]
    explanation_fn: Optional[Callable[[str, str], str]] = field(
        default=None, repr=False
    )
    whole_word: bool = False
    enabled: bool = True

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def explain(self, paragraph: str, matched_phrase: str) -> str:
        """Return an explanation for the detected risk.

        Uses :attr:`explanation_fn` when provided; falls back to a generic
        template otherwise.
        """
        if self.explanation_fn is not None:
            return self.explanation_fn(paragraph, matched_phrase)
        return (
            f"The phrase '{matched_phrase}' in this clause indicates a potential "
            f"{self.category.value} risk. Review this clause with legal counsel."
        )


# ---------------------------------------------------------------------------
# Explanation templates (one per category)
# ---------------------------------------------------------------------------


def _unlimited_liability_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' indicates that liability has not been capped. "
        "Unlimited liability clauses can expose a party to catastrophic financial "
        "risk. Consider negotiating a liability cap (e.g., limited to contract value)."
    )


def _missing_termination_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' suggests an ambiguous or missing termination "
        "mechanism. Without a clear termination clause either party may be locked "
        "into the contract indefinitely or face disputes about exit conditions."
    )


def _one_sided_indemnity_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' creates a one-sided indemnity obligation. "
        "One-sided indemnity clauses require one party to absorb all legal costs "
        "and damages without reciprocal protection. Seek mutual indemnity language."
    )


def _payment_risk_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' signals adverse payment terms. "
        "Conditions such as non-refundable payments, immediate payment demands, or "
        "absence of dispute mechanisms can create significant financial exposure."
    )


def _jurisdiction_risk_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' indicates that disputes may be governed by a "
        "foreign or unfavourable jurisdiction. This can increase litigation costs "
        "and limit access to familiar legal remedies. Negotiate for a neutral forum."
    )


def _confidentiality_risk_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' may introduce inadequate or one-sided "
        "confidentiality obligations. Overly broad NDAs or weak confidentiality "
        "protections can lead to data leaks or unenforceable restrictions."
    )


def _ip_ownership_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' suggests that intellectual property ownership "
        "rights may be transferred or disputed. Ambiguous IP clauses can result "
        "in the loss of created works or proprietary technology. Clarify ownership."
    )


def _penalty_clause_explanation(paragraph: str, phrase: str) -> str:
    return (
        f"The phrase '{phrase}' introduces a penalty or liquidated damages clause. "
        "Such clauses impose predetermined financial consequences for non-performance "
        "and may be disproportionate or unenforceable depending on the jurisdiction."
    )


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

RISK_RULES: List[RiskRule] = [
    # ------------------------------------------------------------------
    # 1. Unlimited Liability
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.UNLIMITED_LIABILITY,
        severity=RiskSeverity.HIGH,
        keywords=[
            "unlimited liability",
            "no liability cap",
            "no cap on liability",
            "liable for all damages",
            "liable for any and all",
            "full liability",
            "not limited in any way",
            "without limitation of liability",
        ],
        explanation_fn=_unlimited_liability_explanation,
    ),
    # ------------------------------------------------------------------
    # 2. Missing / Ambiguous Termination Clause
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.MISSING_TERMINATION,
        severity=RiskSeverity.MEDIUM,
        keywords=[
            "no termination",
            "cannot be terminated",
            "irrevocable",
            "perpetual and irrevocable",
            "no right to terminate",
            "without the right to cancel",
            "non-cancellable",
            "noncancellable",
        ],
        explanation_fn=_missing_termination_explanation,
    ),
    # ------------------------------------------------------------------
    # 3. One-sided Indemnity
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.ONE_SIDED_INDEMNITY,
        severity=RiskSeverity.HIGH,
        keywords=[
            "indemnify and hold harmless",
            "indemnify the client",
            "indemnify the company",
            "defend against all claims",
            "defend, indemnify",
            "hold harmless against all",
            "indemnify against all losses",
            "indemnify against any claim",
            "shall indemnify",
        ],
        explanation_fn=_one_sided_indemnity_explanation,
    ),
    # ------------------------------------------------------------------
    # 4. Payment Risk
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.PAYMENT_RISK,
        severity=RiskSeverity.MEDIUM,
        keywords=[
            "payment due immediately",
            "immediate payment",
            "non-refundable",
            "nonrefundable",
            "no refund",
            "no refunds",
            "advance payment",
            "upfront payment",
            "payment in advance",
            "waives any right to a refund",
            "all fees are final",
        ],
        explanation_fn=_payment_risk_explanation,
    ),
    # ------------------------------------------------------------------
    # 5. Jurisdiction Risk
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.JURISDICTION_RISK,
        severity=RiskSeverity.MEDIUM,
        keywords=[
            "governed by the laws of",
            "exclusive jurisdiction",
            "courts of",
            "arbitration in",
            "venue shall be",
            "subject to the laws of",
            "dispute resolution in",
            "laws of a foreign",
            "international arbitration",
        ],
        explanation_fn=_jurisdiction_risk_explanation,
    ),
    # ------------------------------------------------------------------
    # 6. Confidentiality Risk
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.CONFIDENTIALITY_RISK,
        severity=RiskSeverity.MEDIUM,
        keywords=[
            "perpetual confidentiality",
            "indefinite nda",
            "confidential information shall include all",
            "disclose to any third party",
            "broad confidentiality",
            "no exceptions to confidentiality",
            "all information is confidential",
            "without any restrictions on disclosure",
        ],
        explanation_fn=_confidentiality_risk_explanation,
    ),
    # ------------------------------------------------------------------
    # 7. Intellectual Property Ownership Risk
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.IP_OWNERSHIP_RISK,
        severity=RiskSeverity.HIGH,
        keywords=[
            "intellectual property shall vest",
            "all ip owned by",
            "assigns all rights",
            "work made for hire",
            "waives all moral rights",
            "irrevocably assigns",
            "ownership of all inventions",
            "assigns intellectual property",
            "all developments belong to",
        ],
        explanation_fn=_ip_ownership_explanation,
    ),
    # ------------------------------------------------------------------
    # 8. Penalty Clauses
    # ------------------------------------------------------------------
    RiskRule(
        category=RiskCategory.PENALTY_CLAUSE,
        severity=RiskSeverity.MEDIUM,
        keywords=[
            "liquidated damages",
            "penalty clause",
            "pay a penalty",
            "financial penalty",
            "punitive damages",
            "penalty of",
            "automatic fine",
            "late payment penalty",
            "damages for breach",
        ],
        explanation_fn=_penalty_clause_explanation,
    ),
]

# Convenience lookup by category (populated at import time)
RULES_BY_CATEGORY: dict[RiskCategory, RiskRule] = {
    rule.category: rule for rule in RISK_RULES
}
