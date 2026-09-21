from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FAILURE_TYPES = [
    "ocr_recognition",
    "target_localization",
    "number_text_binding",
    "exact_string_confusion",
    "question_understanding",
    "visual_spatial_reasoning",
    "ambiguity_or_label_issue",
    "other",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a review table for validation errors."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.csv, encoding="utf-8-sig")
    pred = pd.read_csv(args.predictions, encoding="utf-8-sig")

    keep_pred = [c for c in [
        "id", "prediction", "raw_output", "answer", "correct", "category", "latency_seconds"
    ] if c in pred.columns]
    merged = source.merge(pred[keep_pred], on="id", how="inner", suffixes=("", "_pred"))

    if "correct" in merged.columns:
        errors = merged[merged["correct"] == False].copy()  # noqa: E712
    else:
        answer_col = "answer_pred" if "answer_pred" in merged.columns else "answer"
        errors = merged[merged["prediction"] != merged[answer_col]].copy()

    cols = [
        c for c in [
            "id", "path", "category", "question", "a", "b", "c", "d",
            "answer", "prediction", "raw_output"
        ] if c in errors.columns
    ]
    errors = errors[cols].copy()

    errors["failure_type"] = ""
    errors["text_visible_to_human"] = ""
    errors["correct_text_present_in_image"] = ""
    errors["likely_fix"] = ""
    errors["review_notes"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"wrote {len(errors)} errors -> {args.output}")
    print("failure_type options:")
    for item in FAILURE_TYPES:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
