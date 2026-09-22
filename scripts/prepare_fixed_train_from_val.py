from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed training complement from either a 6,714-row "
            "split-assignment CSV with split=train/val or a validation-only CSV."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-val-rows", type=int, default=1007)
    parser.add_argument("--expected-train-rows", type=int, default=5707)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    split_file = pd.read_csv(args.val_csv, encoding="utf-8-sig")

    if "id" not in train.columns or "id" not in split_file.columns:
        raise ValueError("Both train CSV and split/validation CSV must contain an id column.")

    train_ids = train["id"].astype(str)
    if train_ids.duplicated().any():
        raise ValueError("train.csv contains duplicate IDs; cannot safely build complement.")

    split_ids = split_file["id"].astype(str)
    if split_ids.duplicated().any():
        raise ValueError("Split/validation CSV contains duplicate IDs; cannot safely build complement.")

    if "split" in split_file.columns:
        labels = split_file["split"].astype(str).str.strip().str.lower()
        allowed = set(labels.unique())
        if not allowed.issubset({"train", "val"}):
            raise ValueError(f"Unexpected split labels: {sorted(allowed)}")
        val_ids = set(split_file.loc[labels.eq("val"), "id"].astype(str))
        assignment_train_ids = set(split_file.loc[labels.eq("train"), "id"].astype(str))

        if len(val_ids) != args.expected_val_rows:
            raise ValueError(
                f"Split assignment contains {len(val_ids)} val rows, "
                f"expected {args.expected_val_rows}."
            )
        if len(assignment_train_ids) != args.expected_train_rows:
            raise ValueError(
                f"Split assignment contains {len(assignment_train_ids)} train rows, "
                f"expected {args.expected_train_rows}."
            )

        assignment_ids = val_ids | assignment_train_ids
        source_ids = set(train_ids)
        if assignment_ids != source_ids:
            raise ValueError(
                "Split assignment ID set does not match train.csv: "
                f"split_only={len(assignment_ids - source_ids)}, "
                f"train_only={len(source_ids - assignment_ids)}"
            )
    else:
        if len(split_file) != args.expected_val_rows:
            raise ValueError(
                f"Validation-only CSV has {len(split_file)} rows, "
                f"expected {args.expected_val_rows}."
            )
        val_ids = set(split_ids)

    out = train[~train_ids.isin(val_ids)].copy()

    overlap = set(out["id"].astype(str)) & val_ids
    if overlap:
        raise RuntimeError(f"Train/VAL-A ID overlap remains: {len(overlap)}")

    if len(out) != args.expected_train_rows:
        raise RuntimeError(
            f"Unexpected output row count: got {len(out)}, "
            f"expected {args.expected_train_rows}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print({
        "source_train_rows": len(train),
        "split_file_rows": len(split_file),
        "val_unique_ids": len(val_ids),
        "output_train_rows": len(out),
        "overlap": len(overlap),
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
