from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PERCEPTION_FAILURES = {
    "ocr_recognition",
    "exact_string_confusion",
    "target_localization",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a B2 scene-text diagnostic: perception errors + matched correct controls."
    )
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--reviewed-errors", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("splits/b2_scene_text_diagnostic.csv"),
    )
    parser.add_argument(
        "--baseline-subset-predictions",
        type=Path,
        default=Path("output/diagnostics/B0_b2_scene_text_diagnostic_predictions.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/diagnostics/B2_scene_text_diagnostic_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    val = pd.read_csv(args.validation_csv, encoding="utf-8-sig")
    pred = pd.read_csv(args.baseline_predictions, encoding="utf-8-sig")
    review = pd.read_csv(args.reviewed_errors, encoding="utf-8-sig")

    for name, frame in [("validation", val), ("predictions", pred), ("review", review)]:
        if "id" not in frame.columns:
            raise ValueError(f"{name} must contain id")

    required_review = {"failure_type", "category"}
    missing = required_review - set(review.columns)
    if missing:
        raise ValueError(f"review missing columns: {sorted(missing)}")

    pred_key = pred.copy()
    pred_key["_id_key"] = pred_key["id"].astype(str)
    val_key = val.copy()
    val_key["_id_key"] = val_key["id"].astype(str)
    review_key = review.copy()
    review_key["_id_key"] = review_key["id"].astype(str)

    perception = review_key[
        (review_key["category"].astype(str) == "scene_text")
        & (review_key["failure_type"].isin(PERCEPTION_FAILURES))
    ].copy()

    if perception.empty:
        raise ValueError("No scene_text perception errors found in reviewed CSV.")

    correct_scene = pred_key[
        (pred_key["category"].astype(str) == "scene_text")
        & (pred_key["prediction"].astype(str) == pred_key["answer"].astype(str))
    ].copy()

    error_ids = set(perception["_id_key"])
    correct_scene = correct_scene[~correct_scene["_id_key"].isin(error_ids)]

    n_errors = len(perception)
    if len(correct_scene) < n_errors:
        raise ValueError(
            f"Not enough correct scene_text controls: need={n_errors}, available={len(correct_scene)}"
        )

    controls = correct_scene.sample(n=n_errors, random_state=args.seed).copy()

    failure_map = dict(zip(perception["_id_key"], perception["failure_type"].astype(str)))
    selected_ids = list(perception["_id_key"]) + list(controls["_id_key"])
    selected_set = set(selected_ids)

    diagnostic = val_key[val_key["_id_key"].isin(selected_set)].copy()
    diagnostic["diagnostic_group"] = diagnostic["_id_key"].map(
        lambda x: "perception_error" if x in error_ids else "correct_control"
    )
    diagnostic["failure_type"] = diagnostic["_id_key"].map(failure_map).fillna("")
    diagnostic = diagnostic.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    order = {key: i for i, key in enumerate(diagnostic["_id_key"].tolist())}
    baseline_subset = pred_key[pred_key["_id_key"].isin(selected_set)].copy()
    baseline_subset["_order"] = baseline_subset["_id_key"].map(order)
    baseline_subset = baseline_subset.sort_values("_order").drop(columns=["_order"])

    diagnostic = diagnostic.drop(columns=["_id_key"])
    baseline_subset = baseline_subset.drop(columns=["_id_key"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_subset_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    diagnostic.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    baseline_subset.to_csv(
        args.baseline_subset_predictions,
        index=False,
        encoding="utf-8-sig",
    )

    failure_counts = (
        perception["failure_type"].value_counts().sort_index().astype(int).to_dict()
    )
    manifest = {
        "diagnostic_rows": int(len(diagnostic)),
        "perception_error_rows": int(n_errors),
        "correct_control_rows": int(n_errors),
        "perception_failure_counts": failure_counts,
        "seed": args.seed,
        "output_csv": str(args.output_csv),
        "baseline_subset_predictions": str(args.baseline_subset_predictions),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
