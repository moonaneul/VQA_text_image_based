from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Override selected baseline predictions with specialist predictions."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission-output", type=Path)
    args = parser.parse_args()

    base = pd.read_csv(args.baseline, encoding="utf-8-sig")
    spec = pd.read_csv(args.specialist, encoding="utf-8-sig")

    for name, frame in [("baseline", base), ("specialist", spec)]:
        if "id" not in frame.columns or "prediction" not in frame.columns:
            raise ValueError(f"{name} must contain id and prediction")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ids")

    base_key = base["id"].astype(str)
    spec_key = spec["id"].astype(str)
    missing = set(spec_key) - set(base_key)
    if missing:
        raise ValueError(f"{len(missing)} specialist ids are absent from baseline")

    spec_map = dict(zip(spec_key, spec["prediction"]))
    override_mask = base_key.isin(spec_map)
    before = base.loc[override_mask, "prediction"].copy()
    base.loc[override_mask, "prediction"] = base_key[override_mask].map(spec_map).values
    changed = int((before.values != base.loc[override_mask, "prediction"].values).sum())

    if "answer" in base.columns:
        labeled = base["answer"].isin(list("abcd"))
        if labeled.any():
            base.loc[labeled, "correct"] = (
                base.loc[labeled, "prediction"] == base.loc[labeled, "answer"]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.submission_output:
        args.submission_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "id": base["id"],
            "answer": base["prediction"],
        }).to_csv(args.submission_output, index=False, encoding="utf-8-sig")

    result = {
        "baseline_rows": int(len(base)),
        "specialist_rows": int(len(spec)),
        "overridden_rows": int(override_mask.sum()),
        "prediction_changes": changed,
        "output": str(args.output),
        "submission_output": str(args.submission_output) if args.submission_output else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
