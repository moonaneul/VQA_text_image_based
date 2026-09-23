from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = ["team", "perm", "b3", "ft"]


def percentile_rank(s: pd.Series) -> pd.Series:
    # Empirical percentile in [0, 1], stable enough for cross-model comparison
    # without assuming probability calibration is comparable across models.
    return s.rank(method="average", pct=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Apply a calibration-robust selective override gate to previously "
            "computed 3-of-4 consensus diagnostics. Each model's confidence "
            "and margin are converted to within-model empirical percentiles."
        )
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--support-margin-percentile", type=float, default=0.65)
    p.add_argument("--dissent-margin-percentile", type=float, default=0.35)
    p.add_argument("--support-confidence-percentile", type=float, default=0.60)
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    required = {
        "id", "current_best", "diverse_consensus_answer", "selective_override",
        *[f"pred_{m}" for m in MODELS],
        *[f"conf_{m}" for m in MODELS],
        *[f"margin_{m}" for m in MODELS],
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for m in MODELS:
        df[f"conf_pct_{m}"] = percentile_rank(df[f"conf_{m}"].astype(float))
        df[f"margin_pct_{m}"] = percentile_rank(df[f"margin_{m}"].astype(float))

    candidates = df[df["selective_override"] == True].copy()

    support_margin_pct = []
    support_conf_pct = []
    dissent_margin_pct = []
    dissenter = []

    for _, row in candidates.iterrows():
        target = row["diverse_consensus_answer"]
        supporters = [m for m in MODELS if row[f"pred_{m}"] == target]
        dissenters = [m for m in MODELS if row[f"pred_{m}"] != target]
        if len(supporters) != 3 or len(dissenters) != 1:
            support_margin_pct.append(float("nan"))
            support_conf_pct.append(float("nan"))
            dissent_margin_pct.append(float("nan"))
            dissenter.append("+".join(dissenters))
            continue
        d = dissenters[0]
        dissenter.append(d)
        support_margin_pct.append(
            sum(float(row[f"margin_pct_{m}"]) for m in supporters) / 3.0
        )
        support_conf_pct.append(
            sum(float(row[f"conf_pct_{m}"]) for m in supporters) / 3.0
        )
        dissent_margin_pct.append(float(row[f"margin_pct_{d}"]))

    candidates["dissenter"] = dissenter
    candidates["support_margin_pct_mean"] = support_margin_pct
    candidates["support_conf_pct_mean"] = support_conf_pct
    candidates["dissent_margin_pct"] = dissent_margin_pct

    candidates["percentile_gate"] = (
        (candidates["support_margin_pct_mean"] >= args.support_margin_percentile)
        & (candidates["support_conf_pct_mean"] >= args.support_confidence_percentile)
        & (candidates["dissent_margin_pct"] <= args.dissent_margin_percentile)
    )

    selected = candidates[candidates["percentile_gate"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates.to_csv(
        args.output_dir / "all_47_with_percentiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        args.output_dir / "selected_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "rows": int(len(df)),
        "initial_selective_candidates": int(len(candidates)),
        "gate": {
            "support_margin_percentile_mean_min": args.support_margin_percentile,
            "support_confidence_percentile_mean_min": args.support_confidence_percentile,
            "dissent_margin_percentile_max": args.dissent_margin_percentile,
        },
        "selected_candidates": int(len(selected)),
        "dissenter_counts_selected": {
            str(k): int(v) for k, v in selected["dissenter"].value_counts().items()
        },
        "selected_ids": selected["id"].tolist(),
        "transitions": (
            selected.groupby(["current_best", "diverse_consensus_answer"])
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
