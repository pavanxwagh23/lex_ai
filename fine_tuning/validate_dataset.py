"""
Validate the legal LLM fine-tuning dataset before training.

The checks catch common quality problems in generated data, such as duplicate
prompts, malformed rows, law mismatches, and suspicious scenario/law pairings.

Usage:
    python fine_tuning/validate_dataset.py
    python fine_tuning/validate_dataset.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "fine_tuning" / "data" / "legal_qa_dataset.jsonl"

PROMPT_LAW_RE = re.compile(r"\bunder\s+([^:]+?)(?:\s*:\s*|$)", re.IGNORECASE)

SCENARIO_LAW_HINTS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\b(unauthorized access|hacking|bank account|data breach|leaks? confidential)\b", re.I),
        "Information Technology Act, 2000",
        "cyber/data access facts usually need IT Act treatment",
    ),
    (
        re.compile(r"\bcheque\b|\bdishonou?red\b|\binsufficient funds\b", re.I),
        "Negotiable Instruments Act, 1881",
        "cheque dishonour facts usually need NI Act treatment",
    ),
    (
        re.compile(r"\bdefective goods?\b|\bwarranty claims?\b|\bconsumer\b", re.I),
        "Consumer Protection Act, 2019",
        "defective goods/warranty consumer facts usually need consumer law treatment",
    ),
    (
        re.compile(r"\bfails? to perform\b|\bvalid contract\b|\bbreach of contract\b", re.I),
        "Indian Contract Act, 1872",
        "contract performance facts usually need Contract Act treatment",
    ),
]


@dataclass
class Issue:
    line: int
    code: str
    message: str


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    return json.dumps(completion, ensure_ascii=False, sort_keys=True)


def _completion_law(completion: Any) -> str | None:
    if isinstance(completion, dict):
        law = completion.get("law")
        return str(law).strip() if law else None
    return None


def validate_dataset(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    prompts: list[str] = []
    rows: list[tuple[int, dict[str, Any]]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(Issue(line_no, "malformed_json", str(exc)))
                continue

            prompt = str(row.get("prompt", "")).strip()
            completion = row.get("completion")
            completion_text = _completion_to_text(completion).strip()

            if not prompt:
                issues.append(Issue(line_no, "missing_prompt", "Prompt is empty."))
            if not completion_text:
                issues.append(Issue(line_no, "missing_completion", "Completion is empty."))

            if prompt and completion_text:
                prompts.append(_normalise(prompt))
                rows.append((line_no, row))

    duplicate_prompts = {prompt for prompt, count in Counter(prompts).items() if count > 1}
    reported_duplicates: set[str] = set()

    for line_no, row in rows:
        prompt = str(row.get("prompt", "")).strip()
        prompt_norm = _normalise(prompt)
        completion = row.get("completion")
        completion_text = _completion_to_text(completion)

        if prompt_norm in duplicate_prompts and prompt_norm not in reported_duplicates:
            reported_duplicates.add(prompt_norm)
            issues.append(Issue(line_no, "duplicate_prompt", f"Duplicate prompt: {prompt[:90]}"))

        prompt_law_match = PROMPT_LAW_RE.search(prompt)
        prompt_law = prompt_law_match.group(1).strip() if prompt_law_match else None
        completion_law = _completion_law(completion)

        if prompt_law and completion_law and _normalise(prompt_law) != _normalise(completion_law):
            issues.append(
                Issue(
                    line_no,
                    "law_mismatch",
                    f"Prompt asks under '{prompt_law}' but completion law is '{completion_law}'.",
                )
            )

        for pattern, expected_law, reason in SCENARIO_LAW_HINTS:
            if pattern.search(prompt):
                actual = completion_law or completion_text
                if _normalise(expected_law) not in _normalise(actual):
                    issues.append(
                        Issue(
                            line_no,
                            "scenario_law_suspicious",
                            f"Expected '{expected_law}' because {reason}.",
                        )
                    )
                break

        if len(completion_text.split()) < 8:
            issues.append(Issue(line_no, "short_completion", "Completion is very short."))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when issues are found.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Dataset not found: {args.path}")
        return 2

    issues = validate_dataset(args.path)
    print(f"Dataset: {args.path}")
    print(f"Issues found: {len(issues)}")
    for issue in issues[:100]:
        print(f"line {issue.line}: {issue.code}: {issue.message}")
    if len(issues) > 100:
        print(f"... {len(issues) - 100} more issues omitted")

    return 1 if args.strict and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
