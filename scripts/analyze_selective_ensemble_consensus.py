from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]


def load_probs(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={
        "score_a": "p_a",
        "score_b": "p_b",
        "score_c": "p_c",
        "score_d": "p_d",
    })
    required = {"id", *PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    df = df[["id", *PROB_COLS]].copy()
    df[PROB_COLS] = df[PROB_COLS].astype(float)
    sums = df[PROB_COLS].sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"{path} has non-positive score sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)

    prob_matrix = df[PROB_COLS]
    df[f"pred_{name}"] = prob_matrix.idxmax(axis=1).str[-1]
    df[f"conf_{name}"] = prob_matrix.max(axis=1)
    sorted_vals = prob_matrix.apply(lambda row: sorted(row, reverse=True), axis=1)
    df[f"margin_{name}"] = sorted_vals.map(lambda xs: float(xs[0] - xs[1]))

    keep = ["id", f"pred_{name}", f"conf_{name}", f"margin_{name}"]
    return df[keep]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a conservative selective-ensemble rule without using test gold. "
            "Rule candidate: override the current-best answer only when B3P3 and FT-8B "
            "agree on the same alternative and at least one main Qwen3 component "
            "(TEAM-C896 or Perm-8B) agrees with them."
        )
    )
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--perm8b", type=Path, required=True)
    parser.add_argument("--b3p3", type=Path, required=True)
    parser.add_argument("--ft8b", type=Path, required=True)
    parser.add_argument("--current-best", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parts = [
        load_probs(args.team, "team"),
        load_probs(args.perm8b, "perm"),
        load_probs(args.b3p3, "b3"),
        load_probs(args.ft8b, "ft"),
    ]

    m = parts[0]
    for df in parts[1:]:
        m = m.merge(df, on="id", validate="one_to_one")

    current = pd.read_csv(args.current_best, encoding="utf-8-sig")
    if not {"id", "answer"}.issubset(current.columns):
        raise ValueError("current-best must contain id and answer")
    if current["id"].duplicated().any():
        raise ValueError("current-best contains duplicate IDs")
    m = m.merge(
        current[["id", "answer"]].rename(columns={"answer": "current_best"}),
        on="id",
        validate="one_to_one",
    )

    if len(m) != 6714:
        raise ValueError(f"Expected 6714 joined rows, got {len(m)}")

    pred_cols = ["pred_team", "pred_perm", "pred_b3", "pred_ft"]

    def pattern(row):
        counts = pd.Series([row[c] for c in pred_cols]).value_counts()
        vals = sorted(counts.tolist(), reverse=True)
        if vals == [4]:
            return "4-0"
        if vals == [3, 1]:
            return "3-1"
        if vals == [2, 2]:
            return "2-2"
        if vals == [2, 1, 1]:
            return "2-1-1"
        return "1-1-1-1"

    m["agreement_pattern"] = m.apply(pattern, axis=1)

    # General 3-of-4 majority.
    def majority3(row):
        counts = pd.Series([row[c] for c in pred_cols]).value_counts()
        if int(counts.iloc[0]) >= 3:
            return str(counts.index[0])
        return None

    m["majority3_answer"] = m.apply(majority3, axis=1)
    m["majority3_conflict"] = (
        m["majority3_answer"].notna()
        & (m["majority3_answer"] != m["current_best"])
    )

    # Preregistered conservative rule:
    # B3P3 and FT8B must agree, and at least one of TEAM-C896 / Perm8B must join them.
    m["diverse_consensus_answer"] = m["pred_b3"].where(
        (m["pred_b3"] == m["pred_ft"])
        & (
            (m["pred_team"] == m["pred_b3"])
            | (m["pred_perm"] == m["pred_b3"])
        )
    )
    m["selective_override"] = (
        m["diverse_consensus_answer"].notna()
        & (m["diverse_consensus_answer"] != m["current_best"])
    )

    # Useful diagnostics: average confidence/margin among agreeing alternative models.
    m["consensus_support_count"] = (
        (m["pred_team"] == m["diverse_consensus_answer"]).astype(int)
        + (m["pred_perm"] == m["diverse_consensus_answer"]).astype(int)
        + (m["pred_b3"] == m["diverse_consensus_answer"]).astype(int)
        + (m["pred_ft"] == m["diverse_consensus_answer"]).astype(int)
    )

    override = m[m["selective_override"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    m.to_csv(
        args.output_dir / "all_agreement_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    override.to_csv(
        args.output_dir / "selective_override_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "rows": int(len(m)),
        "agreement_patterns": {
            str(k): int(v)
            for k, v in m["agreement_pattern"].value_counts().sort_index().items()
        },
        "majority3_conflicts_with_current_best": int(m["majority3_conflict"].sum()),
        "diverse_consensus_rows": int(m["diverse_consensus_answer"].notna().sum()),
        "selective_override_candidates": int(m["selective_override"].sum()),
        "candidate_answer_transitions": (
            override.groupby(["current_best", "diverse_consensus_answer"])
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
