from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate B2 diagnostic rescues and correct-control regressions."
    )
    parser.add_argument("--diagnostic-csv", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    diag = pd.read_csv(args.diagnostic_csv, encoding="utf-8-sig")
    base = pd.read_csv(args.baseline, encoding="utf-8-sig")
    cand = pd.read_csv(args.candidate, encoding="utf-8-sig")

    for name, frame in [("diagnostic", diag), ("baseline", base), ("candidate", cand)]:
        if "id" not in frame.columns:
            raise ValueError(f"{name} must contain id")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ids")

    required_diag = {"diagnostic_group", "failure_type"}
    missing = required_diag - set(diag.columns)
    if missing:
        raise ValueError(f"diagnostic CSV missing columns: {sorted(missing)}")

    merged = (
        diag[["id", "diagnostic_group", "failure_type"]]
        .merge(
            base[["id", "prediction", "answer"]].rename(
                columns={
                    "prediction": "prediction_baseline",
                    "answer": "answer_baseline",
                }
            ),
            on="id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            cand[["id", "prediction", "answer"]].rename(
                columns={
                    "prediction": "prediction_candidate",
                    "answer": "answer_candidate",
                }
            ),
            on="id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(merged) != len(diag):
        raise ValueError(
            f"ID mismatch: diagnostic={len(diag)}, merged={len(merged)}"
        )

    if not (
        merged["answer_baseline"].astype(str)
        == merged["answer_candidate"].astype(str)
    ).all():
        raise ValueError("Ground-truth answers differ.")

    merged["answer"] = merged["answer_baseline"].astype(str)
    merged["baseline_correct"] = (
        merged["prediction_baseline"].astype(str) == merged["answer"]
    )
    merged["candidate_correct"] = (
        merged["prediction_candidate"].astype(str) == merged["answer"]
    )

    errors = merged[merged["diagnostic_group"] == "perception_error"].copy()
    controls = merged[merged["diagnostic_group"] == "correct_control"].copy()

    if not (~errors["baseline_correct"]).all():
        raise ValueError("Some perception_error rows are not baseline errors.")
    if not controls["baseline_correct"].all():
        raise ValueError("Some correct_control rows are not baseline-correct.")

    rescued = int(errors["candidate_correct"].sum())
    broken = int((~controls["candidate_correct"]).sum())

    by_failure = {}
    for failure, group in errors.groupby("failure_type"):
        by_failure[str(failure)] = {
            "count": int(len(group)),
            "rescued": int(group["candidate_correct"].sum()),
            "rescue_rate": float(group["candidate_correct"].mean()),
        }

    if rescued >= 8 and broken <= 3:
        decision = "PASS_FULL_SCENE_TEXT"
    elif 5 <= rescued <= 7 and broken <= 3:
        decision = "PROMISING_INSPECT_CHANGED"
    else:
        decision = "STOP_TILING_BRANCH"

    result = {
        "diagnostic_count": int(len(merged)),
        "perception_error_count": int(len(errors)),
        "correct_control_count": int(len(controls)),
        "rescued_errors": rescued,
        "still_wrong_errors": int((~errors["candidate_correct"]).sum()),
        "broken_controls": broken,
        "kept_controls": int(controls["candidate_correct"].sum()),
        "net_diagnostic_gain": rescued - broken,
        "by_failure_type": by_failure,
        "decision": decision,
    }

    changed = merged[
        merged["prediction_baseline"].astype(str)
        != merged["prediction_candidate"].astype(str)
    ].copy()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        changed.to_csv(
            args.output.with_name(args.output.stem + "_changed.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
