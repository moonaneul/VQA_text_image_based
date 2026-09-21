from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(n: int, d: int) -> float:
    return float(n / d * 100) if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize manually labeled root-cause error review."
    )
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=1341)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.reviewed_csv, encoding="utf-8-sig")

    required = ["failure_type", "likely_fix"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    labeled = frame[frame["failure_type"].fillna("").astype(str).str.strip() != ""].copy()
    unlabeled = len(frame) - len(labeled)

    failure_counts = labeled["failure_type"].value_counts()
    fix_counts = labeled["likely_fix"].fillna("").replace("", "unspecified").value_counts()

    failure_summary = []
    for name, count in failure_counts.items():
        failure_summary.append({
            "failure_type": str(name),
            "count": int(count),
            "share_of_labeled_errors_pct": pct(int(count), len(labeled)),
            "max_validation_gain_if_fully_fixed_pp": pct(int(count), args.validation_size),
        })

    fix_summary = []
    for name, count in fix_counts.items():
        fix_summary.append({
            "likely_fix": str(name),
            "count": int(count),
            "share_of_labeled_errors_pct": pct(int(count), len(labeled)),
            "max_validation_gain_if_fully_fixed_pp": pct(int(count), args.validation_size),
        })

    category_failure = {}
    if "category" in labeled.columns:
        table = pd.crosstab(labeled["category"], labeled["failure_type"])
        category_failure = {
            str(idx): {str(col): int(table.loc[idx, col]) for col in table.columns}
            for idx in table.index
        }

    visibility = {}
    for col in ["text_visible_to_human", "correct_text_present_in_image"]:
        if col in labeled.columns:
            visibility[col] = {
                str(k): int(v)
                for k, v in labeled[col].fillna("").replace("", "unspecified").value_counts().items()
            }

    result = {
        "reviewed_csv": str(args.reviewed_csv),
        "total_error_rows": int(len(frame)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(unlabeled),
        "completion_rate": float(len(labeled) / len(frame)) if len(frame) else None,
        "validation_size": int(args.validation_size),
        "failure_summary": failure_summary,
        "likely_fix_summary": fix_summary,
        "category_by_failure_type": category_failure,
        "visibility_summary": visibility,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
