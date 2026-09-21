from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_CATEGORIES = {"price", "phone"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the B1 price+phone validation subset and matching B0 predictions."
    )
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument(
        "--subset-csv",
        type=Path,
        default=Path("splits/val_random_price_phone.csv"),
    )
    parser.add_argument(
        "--baseline-subset-predictions",
        type=Path,
        default=Path("output/diagnostics/B0_random_price_phone_predictions.csv"),
    )
    args = parser.parse_args()

    val = pd.read_csv(args.validation_csv, encoding="utf-8-sig")
    pred = pd.read_csv(args.baseline_predictions, encoding="utf-8-sig")

    required_pred = {"id", "prediction", "answer", "category"}
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"baseline predictions missing columns: {sorted(missing)}")
    if "id" not in val.columns:
        raise ValueError("validation CSV must contain id")

    selected_pred = pred[pred["category"].isin(TARGET_CATEGORIES)].copy()
    selected_pred["_id_key"] = selected_pred["id"].astype(str)

    val["_id_key"] = val["id"].astype(str)
    selected_ids = set(selected_pred["_id_key"])
    subset = val[val["_id_key"].isin(selected_ids)].copy()

    if len(subset) != len(selected_pred):
        raise ValueError(
            "Validation/prediction subset id mismatch: "
            f"validation={len(subset)}, predictions={len(selected_pred)}"
        )

    # Preserve validation order while aligning baseline predictions to it.
    order = {key: i for i, key in enumerate(subset["_id_key"].tolist())}
    selected_pred["_order"] = selected_pred["_id_key"].map(order)
    selected_pred = selected_pred.sort_values("_order")

    subset = subset.drop(columns=["_id_key"])
    selected_pred = selected_pred.drop(columns=["_id_key", "_order"])

    args.subset_csv.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_subset_predictions.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.subset_csv, index=False, encoding="utf-8-sig")
    selected_pred.to_csv(
        args.baseline_subset_predictions,
        index=False,
        encoding="utf-8-sig",
    )

    correct = selected_pred["prediction"] == selected_pred["answer"]
    by_category = {}
    for category, group in selected_pred.groupby("category"):
        group_correct = group["prediction"] == group["answer"]
        by_category[str(category)] = {
            "count": int(len(group)),
            "correct": int(group_correct.sum()),
            "errors": int((~group_correct).sum()),
            "accuracy": float(group_correct.mean()),
        }

    result = {
        "subset_csv": str(args.subset_csv),
        "baseline_subset_predictions": str(args.baseline_subset_predictions),
        "count": int(len(selected_pred)),
        "correct": int(correct.sum()),
        "errors": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "by_category": by_category,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
