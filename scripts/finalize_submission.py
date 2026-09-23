from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

CHOICES = {"a", "b", "c", "d"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the current best submission and write the final submission file."
    )
    parser.add_argument(
        "--current-best",
        type=Path,
        default=Path("output/ENSEMBLE_TEAM722_PERM1805_B3P3_0475_FT8B_05.csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/FINAL_SUBMISSION_095263.csv"),
    )
    args = parser.parse_args()

    current = pd.read_csv(args.current_best, encoding="utf-8-sig")
    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")

    for name, df in [("current-best", current), ("sample-submission", sample)]:
        missing = {"id", "answer"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    if len(sample) != 6714:
        raise ValueError(f"Expected 6714 sample rows, got {len(sample)}")
    if len(current) != 6714:
        raise ValueError(f"Expected 6714 current-best rows, got {len(current)}")

    if sample["id"].duplicated().any():
        raise ValueError("sample submission contains duplicate IDs")
    if current["id"].duplicated().any():
        raise ValueError("current-best submission contains duplicate IDs")

    sample_ids = set(sample["id"])
    current_ids = set(current["id"])
    if sample_ids != current_ids:
        missing = sorted(sample_ids - current_ids)[:10]
        extra = sorted(current_ids - sample_ids)[:10]
        raise ValueError(
            f"ID mismatch. missing from current-best={missing}, extra in current-best={extra}"
        )

    invalid = sorted(set(current["answer"].dropna().astype(str)) - CHOICES)
    if invalid:
        raise ValueError(f"Invalid answer labels: {invalid}")
    if current["answer"].isna().any():
        raise ValueError("current-best contains missing answers")

    answer_map = dict(zip(current["id"], current["answer"].astype(str)))
    final = sample.copy()
    final["answer"] = final["id"].map(answer_map)

    if final["answer"].isna().any():
        raise ValueError("Final submission contains missing answers after ID alignment")
    if not final["answer"].isin(CHOICES).all():
        raise ValueError("Final submission contains invalid answers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("FINAL SUBMISSION READY")
    print(f"path: {args.output}")
    print(f"rows: {len(final)}")
    print(f"columns: {list(final.columns)}")
    print(f"duplicate_ids: {int(final['id'].duplicated().sum())}")
    print(f"missing_answers: {int(final['answer'].isna().sum())}")
    print(f"answer_distribution: {final['answer'].value_counts().sort_index().to_dict()}")
    print(f"sha256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
