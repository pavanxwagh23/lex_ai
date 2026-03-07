"""
risk_utils.py
=============
Pure-function utilities for the Risk Detection Engine.

All helpers in this module are stateless and free of side-effects so they
can be imported and unit-tested independently of the rest of the package.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Tuple

from .risk_types import RiskSeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Return a cleaned, lower-cased version of *text* for pattern matching.

    Steps applied:
    1. Unicode NFKD normalisation (handles fancy quotes, dashes, etc.).
    2. Strip leading / trailing whitespace.
    3. Collapse internal whitespace runs to a single space.
    4. Lower-case the result.

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    str
        Normalised string ready for keyword matching.

    Examples
    --------
    >>> normalize_text("  Unlimited  LIABILITY  ")
    'unlimited liability'
    >>> normalize_text("Non\u2011refundable fee")
    'non-refundable fee'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}")

    # 1. Unicode normalisation (NFKD decomposes special characters)
    text = unicodedata.normalize("NFKD", text)
    # Re-encode to ASCII bytes then back to str, drop un-mappable chars
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 2. Collapse whitespace & strip
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Lower-case
    return text.lower()


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def keyword_match(
    text: str,
    keywords: List[str],
    *,
    whole_word: bool = False,
) -> Optional[str]:
    """Return the first keyword from *keywords* found in *text*, or ``None``.

    The comparison is always case-insensitive (both sides are normalised
    before matching).

    Parameters
    ----------
    text:
        The paragraph / sentence to search in (raw or pre-normalised).
    keywords:
        Ordered list of phrases / keywords to search for.
    whole_word:
        If ``True``, each keyword must appear as a whole word (surrounded by
        word boundaries or punctuation).  Useful for avoiding false positives
        like matching "penalty" inside "penalty box" when you only care about
        legal penalty clauses.

    Returns
    -------
    str | None
        The matched keyword exactly as provided in *keywords*, or ``None`` if
        no match was found.

    Examples
    --------
    >>> keyword_match("Company has unlimited liability.", ["unlimited liability"])
    'unlimited liability'
    >>> keyword_match("No such phrase here.", ["unlimited liability"]) is None
    True
    """
    normalised_text = normalize_text(text)

    for kw in keywords:
        normalised_kw = normalize_text(kw)
        if not normalised_kw:
            continue

        if whole_word:
            pattern = r"(?<!\w)" + re.escape(normalised_kw) + r"(?!\w)"
            if re.search(pattern, normalised_text):
                return kw
        else:
            if normalised_kw in normalised_text:
                return kw

    return None


def keyword_match_all(
    text: str,
    keywords: List[str],
    *,
    whole_word: bool = False,
) -> List[str]:
    """Return **all** keywords from *keywords* found in *text*.

    Parameters
    ----------
    text:
        The paragraph / sentence to search in.
    keywords:
        List of phrases to search for.
    whole_word:
        See :func:`keyword_match`.

    Returns
    -------
    list[str]
        All matching keywords (in order of appearance in *keywords*).
    """
    normalised_text = normalize_text(text)
    matched: List[str] = []

    for kw in keywords:
        normalised_kw = normalize_text(kw)
        if not normalised_kw:
            continue

        if whole_word:
            pattern = r"(?<!\w)" + re.escape(normalised_kw) + r"(?!\w)"
            if re.search(pattern, normalised_text):
                matched.append(kw)
        else:
            if normalised_kw in normalised_text:
                matched.append(kw)

    return matched


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


def risk_severity_calculator(
    base_severity: RiskSeverity,
    *,
    match_count: int = 1,
    escalate_threshold: int = 2,
) -> RiskSeverity:
    """Optionally escalate severity when multiple keywords match in one paragraph.

    The logic is intentionally simple so it is easy to adjust:
    * ``match_count >= escalate_threshold``  → escalate by one level
    * Otherwise → return *base_severity* unchanged.

    Parameters
    ----------
    base_severity:
        The default severity defined in the rule.
    match_count:
        Number of keywords from the rule that matched in the paragraph.
    escalate_threshold:
        Minimum match count to trigger escalation (default: 2).

    Returns
    -------
    RiskSeverity
        Possibly-escalated severity.

    Examples
    --------
    >>> risk_severity_calculator(RiskSeverity.LOW, match_count=3, escalate_threshold=2)
    <RiskSeverity.MEDIUM: 'MEDIUM'>
    >>> risk_severity_calculator(RiskSeverity.HIGH, match_count=3)
    <RiskSeverity.HIGH: 'HIGH'>  # already at maximum
    """
    if match_count < escalate_threshold:
        return base_severity

    _escalation: dict[RiskSeverity, RiskSeverity] = {
        RiskSeverity.LOW: RiskSeverity.MEDIUM,
        RiskSeverity.MEDIUM: RiskSeverity.HIGH,
        RiskSeverity.HIGH: RiskSeverity.HIGH,  # already at ceiling
    }
    escalated = _escalation.get(base_severity, base_severity)
    if escalated != base_severity:
        logger.debug(
            "Severity escalated: %s → %s (match_count=%d)",
            base_severity.value,
            escalated.value,
            match_count,
        )
    return escalated


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


def normalize_score(
    raw_score: float,
    max_possible: float,
    *,
    output_max: float = 10.0,
) -> float:
    """Normalise *raw_score* to the range ``[0, output_max]``.

    Parameters
    ----------
    raw_score:
        The sum of individual risk weights before normalisation.
    max_possible:
        The theoretical maximum raw score (used to calibrate the scale).
    output_max:
        Upper bound of the output range (default: 10.0).

    Returns
    -------
    float
        Score in ``[0, output_max]``, rounded to two decimal places.

    Examples
    --------
    >>> normalize_score(6.0, 12.0)
    5.0
    >>> normalize_score(0.0, 12.0)
    0.0
    """
    if max_possible <= 0:
        return 0.0
    ratio = min(raw_score / max_possible, 1.0)
    return round(ratio * output_max, 2)


# ---------------------------------------------------------------------------
# Truncation helper (for display / logging)
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Return *text* truncated to *max_chars* with an ellipsis suffix."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Paragraph deduplication
# ---------------------------------------------------------------------------


def deduplicate_paragraphs(paragraphs: List[str]) -> Tuple[List[str], List[int]]:
    """Remove duplicate paragraphs while preserving original indices.

    Parameters
    ----------
    paragraphs:
        Raw list of input paragraphs.

    Returns
    -------
    tuple[list[str], list[int]]
        * Deduplicated list of paragraphs.
        * Original indices for each kept paragraph.
    """
    seen: set[str] = set()
    unique: List[str] = []
    indices: List[int] = []

    for idx, para in enumerate(paragraphs):
        key = normalize_text(para)
        if key and key not in seen:
            seen.add(key)
            unique.append(para)
            indices.append(idx)

    return unique, indices
