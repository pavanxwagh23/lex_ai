#!/usr/bin/env python3
"""
example_usage.py
================
Standalone demonstration of the Risk Detection Engine.

Run from the project root::

    python example_usage.py

or with the venv activated::

    python ai_engine/risk_detection/example_usage.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when running this file directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_engine.risk_detection import RiskDetectionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup (optional — shows engine internals at DEBUG level)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Sample contract paragraphs
# ---------------------------------------------------------------------------
SAMPLE_PARAGRAPHS: list[str] = [
    # High risk: unlimited liability
    "Company shall have unlimited liability for any damages caused to the "
    "client, including consequential and indirect damages.",
    # Medium risk: termination ambiguity
    "This agreement is irrevocable and cannot be terminated by either party "
    "under any circumstances.",
    # High risk: one-sided indemnity
    "The vendor shall indemnify the client against all losses, claims, "
    "damages, and legal fees that may arise from the performance of this contract.",
    # Medium risk: payment terms
    "All fees paid under this agreement are non-refundable. Payment is due "
    "immediately upon receipt of invoice.",
    # Medium risk: jurisdiction
    "This agreement is governed by the laws of the Cayman Islands, and any "
    "disputes shall be resolved in the exclusive jurisdiction of its courts.",
    # IP risk
    "The developer irrevocably assigns all intellectual property and inventions "
    "created during the engagement to the client.",
    # Penalty clause
    "In the event of a breach, the defaulting party shall pay liquidated damages "
    "of $50,000 per day of delay.",
    # Clean paragraph (should not trigger any rule)
    "Both parties agree to work collaboratively to resolve any disputes through "
    "good-faith negotiation before initiating formal proceedings.",
]

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


def main() -> None:
    engine = RiskDetectionEngine()

    print("\n" + "=" * 70)
    print("  AI Legal Document Analyzer — Risk Detection Engine")
    print("=" * 70)
    print(f"  Analysing {len(SAMPLE_PARAGRAPHS)} contract paragraphs …\n")

    result = engine.analyze(SAMPLE_PARAGRAPHS)

    # Pretty-print JSON output
    output = result.to_dict()
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "-" * 70)
    print(f"  Overall Risk Score : {result.risk_score:.1f} / 10")
    print(f"  Risks Detected     : {len(result.risks_detected)}")
    print(f"  Paragraphs Flagged : {result.risky_paragraph_count} / {result.total_paragraphs}")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # Demonstrate dynamic rule extension
    # -----------------------------------------------------------------------
    from ai_engine.risk_detection import RiskCategory, RiskRule, RiskSeverity

    custom_rule = RiskRule(
        category=RiskCategory.CONFIDENTIALITY_RISK,
        severity=RiskSeverity.LOW,
        keywords=["may share internally", "internal distribution allowed"],
        explanation_fn=lambda para, phrase: (
            f"'{phrase}' permits broad internal sharing of confidential data "
            "without explicit need-to-know restrictions."
        ),
    )
    engine.add_rule(custom_rule)
    print("[Demo] Custom rule added — engine now has", len(engine.active_rules), "active rules.")

    # -----------------------------------------------------------------------
    # Demonstrate category disabling
    # -----------------------------------------------------------------------
    engine.disable_category(RiskCategory.JURISDICTION_RISK)
    print(
        "[Demo] Jurisdiction Risk disabled — engine now has",
        len(engine.active_rules),
        "active rules.",
    )


if __name__ == "__main__":
    main()
