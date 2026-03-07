"""
ai_engine/risk_detection/__init__.py
=====================================
Public API for the Risk Detection package.

Typical usage
-------------
>>> from ai_engine.risk_detection import RiskDetectionEngine
>>> engine = RiskDetectionEngine()
>>> result = engine.analyze(paragraphs)
>>> print(result.to_dict())
"""

from .risk_engine import RiskDetectionEngine
from .risk_rules import RISK_RULES, RULES_BY_CATEGORY, RiskRule
from .risk_types import (
    EMPTY_RESULT,
    RiskAnalysisResult,
    RiskCategory,
    RiskMatch,
    RiskSeverity,
)
from .risk_utils import (
    deduplicate_paragraphs,
    keyword_match,
    keyword_match_all,
    normalize_score,
    normalize_text,
    risk_severity_calculator,
    truncate_text,
)

__all__ = [
    # Engine
    "RiskDetectionEngine",
    # Rules
    "RiskRule",
    "RISK_RULES",
    "RULES_BY_CATEGORY",
    # Types
    "RiskCategory",
    "RiskSeverity",
    "RiskMatch",
    "RiskAnalysisResult",
    "EMPTY_RESULT",
    # Utils
    "normalize_text",
    "keyword_match",
    "keyword_match_all",
    "normalize_score",
    "risk_severity_calculator",
    "truncate_text",
    "deduplicate_paragraphs",
]
