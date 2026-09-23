from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def qstats(s: pd.Series) -> dict:
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize selective-ensemble override candidates without test labels."
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    needed = {
        "id","current_best","diverse_consensus_answer","selective_override",
        "pred_team","pred_perm","pred_b3","pred_ft",
        "conf_team","conf_perm","conf_b3","conf_ft",
        "margin_team","margin_perm","margin_b3","margin_ft",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    c = df[df["selective_override"] == True].copy()
    if c.empty:
        raise ValueError("No selective override candidates found")

    model_cols = ["team","perm","b3","ft"]

    def dissenter(row):
        target = row["diverse_consensus_answer"]
        ds = [m for m in model_cols if row[f"pred_{m}"] != target]
        return ds[0] if len(ds) == 1 else "+".join(ds)

    c["dissenter"] = c.apply(dissenter, axis=1)
    c["current_best_is_dissenter"] = [
        row["current_best"] == row[f"pred_{row['dissenter']}"]
        if row["dissenter"] in model_cols else False
        for _, row in c.iterrows()
    ]

    # Aggregate support confidence/margin across the three consensus voters.
    support_conf = []
    support_margin = []
    dissent_conf = []
    dissent_margin = []
    for _, row in c.iterrows():
        target = row["diverse_consensus_answer"]
        supporters = [m for m in model_cols if row[f"pred_{m}"] == target]
        ds = [m for m in model_cols if row[f"pred_{m}"] != target]
        support_conf.append(sum(float(row[f"conf_{m}"]) for m in supporters) / len(supporters))
        support_margin.append(sum(float(row[f"margin_{m}"]) for m in supporters) / len(supporters))
        if len(ds) == 1:
            d = ds[0]
            dissent_conf.append(float(row[f"conf_{d}"]))
            dissent_margin.append(float(row[f"margin_{d}"]))
        else:
            dissent_conf.append(float("nan"))
            dissent_margin.append(float("nan"))

    c["support_conf_mean"] = support_conf
    c["support_margin_mean"] = support_margin
    c["dissent_conf"] = dissent_conf
    c["dissent_margin"] = dissent_margin
    c["support_minus_dissent_conf"] = c["support_conf_mean"] - c["dissent_conf"]
    c["support_minus_dissent_margin"] = c["support_margin_mean"] - c["dissent_margin"]

    summary = {
        "candidates": int(len(c)),
        "dissenter_counts": {str(k): int(v) for k, v in c["dissenter"].value_counts().items()},
        "current_best_is_dissenter": int(c["current_best_is_dissenter"].sum()),
        "support_conf_mean": qstats(c["support_conf_mean"]),
        "dissent_conf": qstats(c["dissent_conf"].dropna()),
        "support_minus_dissent_conf": qstats(c["support_minus_dissent_conf"].dropna()),
        "support_margin_mean": qstats(c["support_margin_mean"]),
        "dissent_margin": qstats(c["dissent_margin"].dropna()),
        "support_minus_dissent_margin": qstats(c["support_minus_dissent_margin"].dropna()),
    }

    ranked = c.sort_values(
        ["support_minus_dissent_margin","support_minus_dissent_conf"],
        ascending=False,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        ranked.to_csv(
            args.output.with_name(args.output.stem + "_ranked.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop 10 by support-minus-dissent margin:")
    print(
        ranked[
            [
                "id","current_best","diverse_consensus_answer","dissenter",
                "support_conf_mean","dissent_conf","support_minus_dissent_conf",
                "support_margin_mean","dissent_margin","support_minus_dissent_margin",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
