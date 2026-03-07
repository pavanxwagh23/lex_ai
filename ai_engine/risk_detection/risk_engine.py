"""
risk_engine.py
==============
Main Risk Detection Engine for the AI Legal Document Analyzer.

Usage
-----
>>> from ai_engine.risk_detection import RiskDetectionEngine
>>> engine = RiskDetectionEngine()
>>> result = engine.analyze([
...     "Company shall have unlimited liability for any damages caused.",
...     "Either party may terminate the agreement with 30 days notice.",
...     "The vendor shall indemnify the client against all losses.",
... ])
>>> import json
>>> print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from .risk_rules import RISK_RULES, RiskRule
from .risk_types import (
    RiskAnalysisResult,
    RiskCategory,
    RiskMatch,
    RiskSeverity,
)
from .risk_utils import (
    deduplicate_paragraphs,
    keyword_match_all,
    normalize_score,
    risk_severity_calculator,
    truncate_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

# Maximum raw score used for normalisation (adjust when adding many rules)
_MAX_RAW_SCORE: float = 30.0

# Severity → numeric weight (mirrors RiskSeverity.score)
_SEVERITY_WEIGHT: dict[RiskSeverity, int] = {
    RiskSeverity.LOW: 1,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.HIGH: 3,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RiskDetectionEngine:
    """Rule-based engine that detects legal risks in contract paragraphs.

    The engine iterates over every paragraph in the input, applies each
    enabled :class:`~risk_rules.RiskRule`, accumulates findings, and then
    calculates a normalised overall risk score.

    Parameters
    ----------
    rules:
        List of :class:`~risk_rules.RiskRule` objects to apply.  Defaults to
        the built-in :data:`~risk_rules.RISK_RULES` list.
    max_raw_score:
        Theoretical maximum raw score used to normalise the output to 0–10.
        Increase when adding many new rules.
    deduplicate:
        If ``True`` (default), duplicate paragraphs are silently skipped
        before analysis.

    Examples
    --------
    >>> engine = RiskDetectionEngine()
    >>> result = engine.analyze(["Company shall have unlimited liability."])
    >>> result.risk_score
    1.0   # example value — depends on rule set
    """

    def __init__(
        self,
        rules: Optional[List[RiskRule]] = None,
        max_raw_score: float = _MAX_RAW_SCORE,
        deduplicate: bool = True,
    ) -> None:
        self._rules: List[RiskRule] = [r for r in (rules or RISK_RULES) if r.enabled]
        self._max_raw_score = max_raw_score
        self._deduplicate = deduplicate

        logger.info(
            "RiskDetectionEngine initialised with %d active rules.", len(self._rules)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, paragraphs: List[str]) -> RiskAnalysisResult:
        """Analyse a list of contract paragraphs and return a risk report.

        Parameters
        ----------
        paragraphs:
            Raw contract paragraphs.  Empty strings and duplicates are
            handled automatically.

        Returns
        -------
        RiskAnalysisResult
            Structured result containing:
            * overall ``risk_score`` (0–10)
            * list of :class:`~risk_types.RiskMatch` findings
            * paragraph-level statistics

        Raises
        ------
        TypeError
            If *paragraphs* is not a list of strings.
        """
        if not isinstance(paragraphs, list):
            raise TypeError(
                f"'paragraphs' must be a list of strings, got {type(paragraphs).__name__!r}"
            )

        total_paragraphs = len(paragraphs)

        # Optionally deduplicate
        if self._deduplicate:
            paragraphs, original_indices = deduplicate_paragraphs(paragraphs)
        else:
            original_indices = list(range(len(paragraphs)))

        all_matches: List[RiskMatch] = []
        risky_indices: set[int] = set()

        for local_idx, paragraph in enumerate(paragraphs):
            original_idx = original_indices[local_idx]
            para_matches = self._analyze_paragraph(paragraph, original_idx)
            if para_matches:
                risky_indices.add(original_idx)
                all_matches.extend(para_matches)

        # Sort: HIGH first, then MEDIUM, then LOW; stable within same severity
        severity_order = {
            RiskSeverity.HIGH: 0,
            RiskSeverity.MEDIUM: 1,
            RiskSeverity.LOW: 2,
        }
        all_matches.sort(key=lambda m: severity_order[m.severity])

        raw_score = sum(_SEVERITY_WEIGHT[m.severity] for m in all_matches)
        normalised = normalize_score(raw_score, self._max_raw_score)

        result = RiskAnalysisResult(
            risk_score=normalised,
            risks_detected=all_matches,
            total_paragraphs=total_paragraphs,
            risky_paragraph_count=len(risky_indices),
        )

        logger.info(
            "Analysis complete: %d risks detected, score=%.2f (%d/%d paragraphs flagged).",
            len(all_matches),
            normalised,
            len(risky_indices),
            total_paragraphs,
        )
        return result

    def analyze_json(self, paragraphs: List[str]) -> str:
        """Convenience method that returns :meth:`analyze` result as a JSON string.

        Parameters
        ----------
        paragraphs:
            See :meth:`analyze`.

        Returns
        -------
        str
            Pretty-printed JSON representation of the analysis result.
        """
        result = self.analyze(paragraphs)
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    def add_rule(self, rule: RiskRule) -> None:
        """Dynamically register a new rule at runtime.

        Parameters
        ----------
        rule:
            A configured :class:`~risk_rules.RiskRule` instance.
        """
        if not rule.enabled:
            logger.warning("Rule for %s is disabled and will not be applied.", rule.category)
            return
        self._rules.append(rule)
        logger.info("Rule added: %s (severity=%s)", rule.category.value, rule.severity.value)

    def disable_category(self, category: RiskCategory) -> None:
        """Remove all rules matching *category* from the active rule set.

        Parameters
        ----------
        category:
            The :class:`~risk_types.RiskCategory` to silence.
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.category != category]
        removed = before - len(self._rules)
        logger.info(
            "Disabled %d rule(s) for category '%s'.", removed, category.value
        )

    @property
    def active_rules(self) -> List[RiskRule]:
        """Return a read-only view of the currently active rules."""
        return list(self._rules)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_paragraph(
        self, paragraph: str, paragraph_index: int
    ) -> List[RiskMatch]:
        """Apply all rules to a single paragraph and return matches.

        Parameters
        ----------
        paragraph:
            Raw paragraph text.
        paragraph_index:
            Original zero-based index in the caller's input list.

        Returns
        -------
        list[RiskMatch]
            All risk findings for this paragraph (may be empty).
        """
        if not paragraph or not paragraph.strip():
            return []

        matches: List[RiskMatch] = []

        for rule in self._rules:
            matched_phrases = keyword_match_all(
                paragraph, rule.keywords, whole_word=rule.whole_word
            )
            if not matched_phrases:
                continue

            # Use the first (most-specific longer?) phrase for the display
            primary_match = matched_phrases[0]

            # Optionally escalate severity when many phrases hit simultaneously
            effective_severity = risk_severity_calculator(
                rule.severity,
                match_count=len(matched_phrases),
                escalate_threshold=2,
            )

            explanation = rule.explain(paragraph, primary_match)

            match = RiskMatch(
                category=rule.category,
                severity=effective_severity,
                text=paragraph,
                explanation=explanation,
                matched_phrase=primary_match,
                paragraph_index=paragraph_index,
            )
            matches.append(match)

            logger.debug(
                "Rule fired: [%s/%s] paragraph_index=%d phrase=%r",
                rule.category.value,
                effective_severity.value,
                paragraph_index,
                truncate_text(primary_match, 60),
            )

        return matches
