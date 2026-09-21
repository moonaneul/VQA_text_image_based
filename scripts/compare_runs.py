from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


CATEGORIES = ["scene_text", "phone", "price", "menu", "count", "color", "spatial", "other"]


def prepare(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "prediction", "answer", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids.")

    frame = frame[["id", "prediction", "answer", "category"]].copy()
    frame[f"correct_{suffix}"] = frame["prediction"] == frame["answer"]
    return frame.rename(columns={
        "prediction": f"prediction_{suffix}",
        "answer": f"answer_{suffix}",
        "category": f"category_{suffix}",
    })


def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def summarize(frame: pd.DataFrame) -> dict:
    baseline_correct = frame["correct_baseline"]
    candidate_correct = frame["correct_candidate"]

    old_wrong_new_right = int((~baseline_correct & candidate_correct).sum())
    old_right_new_wrong = int((baseline_correct & ~candidate_correct).sum())
    both_right = int((baseline_correct & candidate_correct).sum())
    both_wrong = int((~baseline_correct & ~candidate_correct).sum())
    disagreement = frame["prediction_baseline"] != frame["prediction_candidate"]

    baseline_accuracy = float(baseline_correct.mean())
    candidate_accuracy = float(candidate_correct.mean())

    return {
        "count": int(len(frame)),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta_accuracy": candidate_accuracy - baseline_accuracy,
        "delta_percentage_points": (candidate_accuracy - baseline_accuracy) * 100,
        "old_wrong_new_right": old_wrong_new_right,
        "old_right_new_wrong": old_right_new_wrong,
        "net_gain_samples": old_wrong_new_right - old_right_new_wrong,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(
            old_wrong_new_right, old_right_new_wrong
        ),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "prediction_disagreement_rate": float(disagreement.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise comparison of two VQA prediction CSV files on identical sample ids."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = prepare(args.baseline, "baseline")
    candidate = prepare(args.candidate, "candidate")
    merged = baseline.merge(candidate, on="id", how="inner", validate="one_to_one")

    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError(
            "Prediction files do not contain the same sample ids. "
            f"baseline={len(baseline)}, candidate={len(candidate)}, overlap={len(merged)}"
        )

    answer_mismatch = merged["answer_baseline"].astype(str) != merged["answer_candidate"].astype(str)
    if answer_mismatch.any():
        raise ValueError("Ground-truth answers differ between the two prediction files.")

    category_mismatch = merged["category_baseline"].astype(str) != merged["category_candidate"].astype(str)
    if category_mismatch.any():
        raise ValueError("Question categories differ between the two prediction files.")

    merged["category"] = merged["category_baseline"]
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "overall": summarize(merged),
        "by_category": {},
    }

    for category in CATEGORIES:
        subset = merged[merged["category"] == category]
        if not subset.empty:
            result["by_category"][category] = summarize(subset)

    changed = merged[merged["prediction_baseline"] != merged["prediction_candidate"]].copy()
    changed_columns = [
        "id", "category", "answer_baseline",
        "prediction_baseline", "prediction_candidate",
        "correct_baseline", "correct_candidate",
    ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        changed.to_csv(
            args.output.with_name(args.output.stem + "_changed.csv"),
            columns=changed_columns,
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
