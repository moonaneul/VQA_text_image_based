from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CHOICES = {"a", "b", "c", "d"}


def summarize(frame: pd.DataFrame) -> dict:
    out = {"count": int(len(frame))}
    if not len(frame):
        return out

    if "answer" in frame.columns and "prediction" in frame.columns:
        labeled = frame[frame["answer"].isin(CHOICES)].copy()
        if len(labeled):
            correct = labeled["prediction"] == labeled["answer"]
            out["accuracy"] = float(correct.mean())
            out["correct"] = int(correct.sum())
            out["errors"] = int((~correct).sum())

    if "category" in frame.columns:
        out["by_category"] = {
            str(k): int(v)
            for k, v in frame["category"].value_counts().sort_index().items()
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze accuracy/category mix for exact vs non-exact generation outputs."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")
    if "raw_output" not in frame.columns:
        raise ValueError("predictions.csv must contain raw_output.")

    raw = frame["raw_output"].fillna("").astype(str).str.strip().str.lower()
    exact_mask = raw.isin(CHOICES)

    exact = frame.loc[exact_mask].copy()
    non_exact = frame.loc[~exact_mask].copy()

    result = {
        "file": str(args.predictions),
        "overall": summarize(frame),
        "exact_one_letter": summarize(exact),
        "non_exact": summarize(non_exact),
        "non_exact_rate": float((~exact_mask).mean()) if len(frame) else None,
    }

    if "answer" in frame.columns and "prediction" in frame.columns:
        for part_name, part in [("exact_one_letter", exact), ("non_exact", non_exact)]:
            if "category" not in part.columns:
                continue
            by_cat = {}
            for category, subset in part.groupby("category"):
                by_cat[str(category)] = summarize(subset)
            result[part_name]["category_detail"] = by_cat

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
