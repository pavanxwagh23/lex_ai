"""
fine_tuning/generate_dataset.py
================================
Generates a labeled legal Q&A fine-tuning dataset from your existing
clause data and contract samples.

Outputs: fine_tuning/data/legal_qa_dataset.jsonl

Each line is a JSON object:
    {"prompt": "...", "completion": "..."}

Usage:
    python fine_tuning/generate_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "fine_tuning" / "data"
OUTPUT_FILE = OUTPUT_DIR / "legal_qa_dataset.jsonl"

# ---------------------------------------------------------------------------
# Seed Q&A pairs — legal domain knowledge base
# ---------------------------------------------------------------------------

_SEED_QA: list[dict] = [
    # Warranty
    {"prompt": "What is a warranty in a contract?",
     "completion": "A warranty is a legally binding promise that certain facts are true. Express warranties are explicitly stated; implied warranties arise automatically by law. A breach of warranty entitles the non-breaching party to damages."},
    {"prompt": "What is the difference between a warranty and a representation?",
     "completion": "A representation is a statement of fact made to induce the other party to enter the contract. A warranty is a contractual promise that the fact remains true throughout the contract. Breaching a representation may allow rescission; breaching a warranty gives rise to a damages claim."},

    # Liability
    {"prompt": "What does limitation of liability mean?",
     "completion": "A limitation of liability clause caps the maximum amount one party can recover from the other for breach. Common caps include total fees paid, 3 months of fees, or a fixed dollar amount. Without a cap, liability is unlimited — a HIGH RISK in any contract."},
    {"prompt": "What is indemnification in a contract?",
     "completion": "Indemnification obligates one party to compensate the other for specified losses, damages, or legal costs. A broad, one-sided indemnification without a liability cap is a significant legal risk."},

    # Termination
    {"prompt": "What is a termination for convenience clause?",
     "completion": "Termination for convenience allows either party to end the contract at any time, without needing to prove a breach. Unlike termination for cause, no wrongdoing is required. Important provisions include required notice periods and payment for work completed."},
    {"prompt": "What happens to IP rights when a contract is terminated?",
     "completion": "When a contract ends, IP ownership depends on whether a license or assignment was granted. Licenses typically end with the contract; assignments are permanent. Always check the survival clause to see which provisions outlast termination."},

    # NDA / Confidentiality
    {"prompt": "How long should an NDA last?",
     "completion": "A standard NDA confidentiality obligation lasts 2–5 years after the contract ends. Perpetual confidentiality obligations are common for trade secrets. Obligations lasting over 10 years for general business information are often considered overly broad."},
    {"prompt": "What information is typically excluded from an NDA?",
     "completion": "Typical exceptions include: information already publicly known, information independently developed by the receiving party, information received from a third party without restriction, and disclosures required by law or court order."},

    # Force Majeure
    {"prompt": "What is a force majeure clause?",
     "completion": "Force majeure excuses a party from performing when extraordinary events beyond their control occur — such as natural disasters, wars, pandemics, or government actions. Always check if the clause is too broad and whether there is an obligation to resume performance after the event ends."},

    # Governing Law / Dispute Resolution
    {"prompt": "Why is the governing law clause important?",
     "completion": "The governing law clause determines which jurisdiction's legal framework applies to the contract. A foreign jurisdiction can dramatically increase legal costs and reduce enforceability of your rights. Always negotiate for your home jurisdiction where possible."},
    {"prompt": "What is the difference between arbitration and litigation?",
     "completion": "Arbitration is a private dispute resolution process — faster, confidential, and typically cheaper than court. Litigation is the public court process with formal discovery, juries, and extensive appeals. Many commercial contracts mandate arbitration to avoid unpredictable juries."},

    # Non-compete
    {"prompt": "Are non-compete agreements enforceable?",
     "completion": "Enforceability varies by jurisdiction. California and North Dakota ban employee non-competes entirely. Most states enforce them only if they are reasonable in duration (under 2 years), geographic scope, and industry breadth. Overly broad non-competes are routinely struck down by courts."},

    # IP
    {"prompt": "What is the work-for-hire doctrine?",
     "completion": "Work-for-hire means that IP created by an employee or contractor belongs to the party who commissioned the work, not the creator. In the US, certain categories of contracted works automatically qualify as work-for-hire; otherwise, an explicit written assignment is required."},

    # Payment
    {"prompt": "What is a Net 30 payment term?",
     "completion": "Net 30 means the full invoice amount is due within 30 days of the invoice date. Net 60 and Net 90 are common alternatives. Many contracts include late payment interest (e.g., 1.5% per month) for overdue amounts. Automatic price escalation clauses can increase rates annually — always check for these."},

    # General
    {"prompt": "What makes a contract legally binding?",
     "completion": "A legally binding contract requires: (1) Offer — one party proposes terms; (2) Acceptance — the other party agrees; (3) Consideration — something of value exchanged; (4) Capacity — both parties have legal standing to contract; (5) Legality — the contract purpose must be lawful."},
    {"prompt": "What is the parol evidence rule?",
     "completion": "The parol evidence rule prevents parties from introducing oral or written evidence made before or during contract signing that contradicts the final written agreement. This is why 'entire agreement' clauses are included in contracts — to prevent such disputes."},
    {"prompt": "What is a severability clause?",
     "completion": "A severability clause states that if one provision is found unenforceable, the remaining provisions continue in full force. Without severability, a single unenforceable clause could potentially void the entire contract."},
    {"prompt": "What is the difference between a void and voidable contract?",
     "completion": "A void contract has no legal effect from the start — it cannot be enforced by either party (e.g., contracts for illegal activities). A voidable contract is initially valid but can be rejected by one party due to misrepresentation, fraud, duress, or incapacity."},
]


def _augment_prompts(qa: dict) -> list[dict]:
    """Create slight prompt variations for data diversity."""
    prompt = qa["prompt"]
    completion = qa["completion"]
    variants = [
        {"prompt": prompt, "completion": completion},
        {"prompt": f"Explain: {prompt}", "completion": completion},
        {"prompt": f"In simple terms, {prompt.lower()}", "completion": completion},
    ]
    return variants


def generate_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    for qa in _SEED_QA:
        all_samples.extend(_augment_prompts(qa))

    random.shuffle(all_samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(all_samples)} Q&A training samples → {OUTPUT_FILE}")
    print("\nSample entry:")
    print(json.dumps(all_samples[0], indent=2))


if __name__ == "__main__":
    generate_dataset()
