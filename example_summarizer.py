#!/usr/bin/env python3
"""
example_summarizer.py
=====================
Standalone demonstration of the LegalSummarizer module.

Run from the project root (with venv activated):

    python example_summarizer.py

NOTE: The first run downloads ~1.6 GB (BART model weights) from HuggingFace.
Subsequent runs use the local cache — startup is under 10 seconds.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_engine.summarizer import LegalSummarizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Sample legal contract text (condensed for demo)
# ---------------------------------------------------------------------------
SAMPLE_CONTRACT = """
MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of January 1, 2025,
by and between Acme Corp, a Delaware corporation ("Client"), and TechVendor Inc.,
a California corporation ("Vendor").

1. SERVICES
Vendor shall provide software development and consulting services as described in
each Statement of Work ("SOW") executed under this Agreement. Vendor must deliver
all services outlined in Schedule A with reasonable care and professional skill.
The Client must provide timely access to necessary resources and personnel.

2. PAYMENT TERMS
Client agrees to pay Vendor's fees as specified in each applicable SOW.
Invoices shall be issued on a monthly basis and are due within 30 days of receipt.
A late payment fee of 1.5% per month will be applied to any overdue balance.
All fees paid under this agreement are non-refundable.
Payment shall be made in US dollars via wire transfer or ACH.

3. TERM AND TERMINATION
This Agreement commences on January 1, 2025, and continues for one (1) year unless
earlier terminated. Either party may terminate this Agreement with 30 days prior
written notice. The Client may terminate immediately if Vendor commits a material
breach not cured within 10 business days of written notice.

4. INTELLECTUAL PROPERTY
All work product, deliverables, and developments created by Vendor under this
Agreement shall be considered works made for hire. Upon full payment, all intellectual
property rights shall be assigned and transferred to Client. Vendor retains no rights
to the deliverables after termination.

5. CONFIDENTIALITY
Each party agrees to maintain in strict confidence all Confidential Information
of the other party. Neither party shall disclose any confidential information to
any third party without prior written consent. This obligation survives termination
for a period of three (3) years.

6. LIABILITY
Vendor's total liability under this Agreement shall not exceed the fees paid in the
three (3) months preceding the claim. In no event shall either party be liable for
indirect, incidental, or consequential damages.

7. INDEMNIFICATION
Vendor shall indemnify and hold harmless the Client against any third-party claims
arising out of Vendor's gross negligence or willful misconduct. Client shall
indemnify Vendor against claims arising from Client-provided materials.

8. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.
Any disputes shall be resolved through binding arbitration in San Francisco, California,
under the rules of the American Arbitration Association.

9. DISPUTE RESOLUTION
The parties agree to attempt in good faith to resolve any dispute through negotiation
before initiating formal arbitration proceedings. The arbitration award shall be final
and binding upon both parties.

10. GENERAL PROVISIONS
This Agreement constitutes the entire agreement between the parties. Any amendment
must be made in writing and signed by both parties. This Agreement may not be
assigned by either party without the prior written consent of the other party.
If any provision of this Agreement is found to be unenforceable, the remaining
provisions shall continue in full force and effect.
"""


def main() -> None:
    print("\n" + "=" * 70)
    print("  AI Legal Document Analyzer — Summarization Demo")
    print("=" * 70)
    print(f"  Contract length : {len(SAMPLE_CONTRACT):,} characters")

    summarizer = LegalSummarizer()

    # ------------------------------------------------------------------
    # Option A: Full transformer-based summary (requires model download)
    # ------------------------------------------------------------------
    use_full_model = "--full" in sys.argv

    if use_full_model:
        print("\n  Mode: FULL (transformer model — may take a few minutes first run)")
        print("-" * 70)
        result = summarizer.generate_summary(SAMPLE_CONTRACT)
    else:
        # ------------------------------------------------------------------
        # Option B: Lightweight demo — shows chunking + rule-based sections
        # (skips the model download so CI / quick demos work without GPU)
        # ------------------------------------------------------------------
        print("\n  Mode: LIGHTWEIGHT (chunking + rule-based sections only)")
        print("  Run with --full to enable the transformer model.\n")
        print("-" * 70)

        from ai_engine.summarizer import SummaryResult, _clean_text, _extract_sections, _sentence_tokenize  # noqa: E402

        cleaned = _clean_text(SAMPLE_CONTRACT)
        sentences = _sentence_tokenize(cleaned)
        sections = _extract_sections(sentences)

        chunks = summarizer.chunk_text(SAMPLE_CONTRACT)
        print(f"  Text split into {len(chunks)} chunk(s).")
        for i, c in enumerate(chunks, 1):
            print(f"    Chunk {i}: {len(c.split())} words")

        # Build a result with extracted sections but a placeholder summary
        result = SummaryResult(
            overall_summary=(
                "This is a Master Services Agreement between Acme Corp (Client) and "
                "TechVendor Inc. (Vendor) governing software development and consulting "
                "services. It sets out payment, IP ownership, confidentiality, liability "
                "limits, indemnification, and California-governed dispute resolution terms."
            ),
            key_points=[
                "Agreement is for software development and consulting services.",
                "Invoices due within 30 days; 1.5% monthly late fee applies.",
                "Either party may terminate with 30 days written notice.",
                "All IP is assigned to Client upon full payment.",
                "Vendor liability capped at 3 months' fees.",
                "Disputes resolved by binding arbitration in San Francisco.",
            ],
            obligations=sections.get("obligations", []),
            payment_terms=sections.get("payment", []),
            termination=sections.get("termination", []),
            risks=sections.get("risk", []),
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print(result.to_text())

    # Also show JSON representation
    print("\n  JSON Output (for FastAPI / API consumers):")
    print("-" * 70)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
