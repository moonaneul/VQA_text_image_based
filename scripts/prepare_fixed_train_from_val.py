from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the fixed training complement by excluding all VAL-A IDs from train.csv."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    val = pd.read_csv(args.val_csv, encoding="utf-8-sig")

    if "id" not in train.columns or "id" not in val.columns:
        raise ValueError("Both train and validation CSVs must contain an id column.")

    val_ids = set(val["id"].astype(str))
    out = train[~train["id"].astype(str).isin(val_ids)].copy()

    overlap = set(out["id"].astype(str)) & val_ids
    if overlap:
        raise RuntimeError(f"Train/VAL-A ID overlap remains: {len(overlap)}")

    expected = len(train) - len(val_ids)
    if len(out) != expected:
        raise RuntimeError(
            f"Unexpected row count: got {len(out)}, expected {expected}. "
            "Check duplicate IDs in train/VAL-A."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print({
        "train_rows": len(train),
        "val_rows": len(val),
        "val_unique_ids": len(val_ids),
        "output_train_rows": len(out),
        "overlap": len(overlap),
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
