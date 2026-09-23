from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]


def load_probs(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {
        "score_a": "p_a",
        "score_b": "p_b",
        "score_c": "p_c",
        "score_d": "p_d",
    }
    df = df.rename(columns=rename)
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
        raise ValueError(f"{path} contains non-positive score sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)
    return df.rename(columns={c: f"{c}_{suffix}" for c in PROB_COLS})


def pred_from_suffix(df: pd.DataFrame, suffix: str) -> pd.Series:
    cols = [f"p_{c}_{suffix}" for c in CHOICES]
    return df[cols].idxmax(axis=1).str.extract(r"p_([abcd])_", expand=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered FT-8B cross-view consensus ensemble candidate. "
            "Keep 640-P0 by default; override the FT slot only when remapped 640-P1 "
            "and 896-P0 agree against 640-P0. Existing ensemble weights are fixed."
        )
    )
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--perm8b", type=Path, required=True)
    parser.add_argument("--b3p3", type=Path, required=True)
    parser.add_argument("--ft-p0-640", type=Path, required=True)
    parser.add_argument("--ft-p1-640", type=Path, required=True)
    parser.add_argument("--ft-p0-896", type=Path, required=True)
    parser.add_argument("--current-best", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    team = load_probs(args.team, "team")
    perm = load_probs(args.perm8b, "perm")
    b3 = load_probs(args.b3p3, "b3")
    p0 = load_probs(args.ft_p0_640, "p0")
    p1 = load_probs(args.ft_p1_640, "p1")
    p896 = load_probs(args.ft_p0_896, "p896")

    merged = team
    for df in [perm, b3, p0, p1, p896]:
        merged = merged.merge(df, on="id", validate="one_to_one")

    if len(merged) != 6714:
        raise ValueError(f"Expected 6714 joined rows, got {len(merged)}")

    merged["pred_p0"] = pred_from_suffix(merged, "p0")
    merged["pred_p1"] = pred_from_suffix(merged, "p1")
    merged["pred_p896"] = pred_from_suffix(merged, "p896")

    merged["override"] = (
        (merged["pred_p1"] == merged["pred_p896"])
        & (merged["pred_p1"] != merged["pred_p0"])
    )

    for c in CHOICES:
        merged[f"p_{c}_ftcons"] = merged[f"p_{c}_p0"]
        mask = merged["override"]
        merged.loc[mask, f"p_{c}_ftcons"] = (
            0.5 * merged.loc[mask, f"p_{c}_p1"]
            + 0.5 * merged.loc[mask, f"p_{c}_p896"]
        )

    # Preserve the current-best ensemble weights exactly.
    weights = {
        "team": 0.7220,
        "perm": 0.1805,
        "b3": 0.0475,
        "ftcons": 0.0500,
    }

    score_cols = []
    for c in CHOICES:
        score = (
            weights["team"] * np.log(merged[f"p_{c}_team"].to_numpy() + 1e-12)
            + weights["perm"] * np.log(merged[f"p_{c}_perm"].to_numpy() + 1e-12)
            + weights["b3"] * np.log(merged[f"p_{c}_b3"].to_numpy() + 1e-12)
            + weights["ftcons"] * np.log(merged[f"p_{c}_ftcons"].to_numpy() + 1e-12)
        )
        col = f"ensemble_logscore_{c}"
        merged[col] = score
        score_cols.append(col)

    labels = np.array(list(CHOICES))
    new_pred = labels[merged[score_cols].to_numpy().argmax(axis=1)]

    current = pd.read_csv(args.current_best, encoding="utf-8-sig")
    if "answer" not in current.columns:
        raise ValueError("current-best submission must contain answer column")
    if current["id"].duplicated().any():
        raise ValueError("current-best submission contains duplicate IDs")
    current_map = dict(zip(current["id"], current["answer"]))
    current_pred = merged["id"].map(current_map)
    if current_pred.isna().any():
        raise ValueError("current-best ID set does not match probability inputs")

    changed_mask = current_pred.to_numpy() != new_pred
    changed_count = int(changed_mask.sum())
    override_count = int(merged["override"].sum())

    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    new_map = dict(zip(merged["id"], new_pred))
    sample["answer"] = sample["id"].map(new_map)
    if sample["answer"].isna().any():
        raise ValueError("sample submission contains IDs missing from ensemble inputs")

    args.output_dir.mkdir(parents=True, exist_ok=False)

    sample.to_csv(
        args.output_dir / "submission.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ft_probs = merged[
        ["id", *[f"p_{c}_ftcons" for c in CHOICES]]
    ].rename(columns={f"p_{c}_ftcons": f"p_{c}" for c in CHOICES})
    ft_probs.to_csv(
        args.output_dir / "ft8b_crossview_choice_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changed = pd.DataFrame({
        "id": merged.loc[changed_mask, "id"],
        "current_best_answer": current_pred[changed_mask].to_numpy(),
        "new_answer": new_pred[changed_mask],
        "ft_p0_answer": merged.loc[changed_mask, "pred_p0"].to_numpy(),
        "ft_p1_answer": merged.loc[changed_mask, "pred_p1"].to_numpy(),
        "ft_896_answer": merged.loc[changed_mask, "pred_p896"].to_numpy(),
        "crossview_override": merged.loc[changed_mask, "override"].to_numpy(),
    })
    changed.to_csv(
        args.output_dir / "changed_vs_current_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overrides = merged.loc[
        merged["override"],
        ["id", "pred_p0", "pred_p1", "pred_p896"],
    ].copy()
    overrides.to_csv(
        args.output_dir / "crossview_overrides.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"rows: {len(merged)}")
    print(f"crossview overrides inside FT slot: {override_count}")
    print(f"changed vs current best: {changed_count}")
    print(f"changed rate: {changed_count / len(merged):.6f}")
    print("new distribution:", pd.Series(new_pred).value_counts().sort_index().to_dict())
    print("submission:", args.output_dir / "submission.csv")
    print("changed rows:", args.output_dir / "changed_vs_current_best.csv")
    print("all FT overrides:", args.output_dir / "crossview_overrides.csv")


if __name__ == "__main__":
    main()
