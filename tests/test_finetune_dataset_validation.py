import json

from fine_tuning.validate_dataset import validate_dataset


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_validate_dataset_flags_law_mismatch(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "prompt": "Analyze the legal scenario under Information Technology Act, 2000: a person gains unauthorized access to another user's bank account.",
                "completion": {"law": "Negotiable Instruments Act, 1881", "sections": ["Section 138"]},
            }
        ],
    )

    issues = validate_dataset(dataset)
    codes = {issue.code for issue in issues}

    assert "law_mismatch" in codes
    assert "scenario_law_suspicious" in codes


def test_validate_dataset_accepts_consistent_row(tmp_path):
    dataset = tmp_path / "good.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "prompt": "Analyze the legal scenario under Negotiable Instruments Act, 1881: a cheque is dishonored due to insufficient funds.",
                "completion": {
                    "law": "Negotiable Instruments Act, 1881",
                    "sections": ["Section 138"],
                    "analysis": "Cheque dishonour may create statutory liability if required ingredients are met.",
                },
            }
        ],
    )

    assert validate_dataset(dataset) == []
