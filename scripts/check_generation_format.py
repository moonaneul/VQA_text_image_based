from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


CHOICES = {"a", "b", "c", "d"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether free-generation outputs are already exact one-letter choices."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")
    if "raw_output" not in frame.columns:
        raise ValueError("predictions.csv must contain raw_output.")

    raw = frame["raw_output"].fillna("").astype(str)
    stripped = raw.str.strip()
    lower = stripped.str.lower()

    exact = lower.isin(CHOICES)
    parsed_choice_like = lower.str.contains(
        r"(?:^|[^a-z])\(?[abcd]\)?(?:[^a-z]|$)",
        regex=True,
    )
    empty = stripped.eq("")

    summary = {
        "file": str(args.predictions),
        "count": int(len(frame)),
        "exact_one_letter": int(exact.sum()),
        "exact_one_letter_rate": float(exact.mean()) if len(frame) else None,
        "non_exact": int((~exact).sum()),
        "empty": int(empty.sum()),
        "choice_like_but_non_exact": int((~exact & parsed_choice_like).sum()),
        "unique_raw_outputs_top20": raw.value_counts(dropna=False).head(20).to_dict(),
    }

    non_exact_rows = frame.loc[~exact].copy()
    keep = [c for c in ["id", "raw_output", "prediction", "answer", "category"] if c in non_exact_rows.columns]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        non_exact_rows[keep].to_csv(
            args.output.with_name(args.output.stem + "_non_exact.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
