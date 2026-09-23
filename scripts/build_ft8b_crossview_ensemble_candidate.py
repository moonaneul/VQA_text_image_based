from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]


def load_probs(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {"score_a":"p_a","score_b":"p_b","score_c":"p_c","score_d":"p_d"}
    df = df.rename(columns=rename)
    required = {"id", *PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    df = df[["id", *PROB_COLS]].copy()
    sums = df[PROB_COLS].sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"{path} has non-positive probability sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)
    return df.rename(columns={c:f"{c}_{suffix}" for c in PROB_COLS})


def argmax_choice(df: pd.DataFrame, suffix: str) -> pd.Series:
    cols = [f"p_{c}_{suffix}" for c in CHOICES]
    return df[cols].idxmax(axis=1).str.extract(r"p_([abcd])_", expand=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team", type=Path, required=True)
    p.add_argument("--perm8b", type=Path, required=True)
    p.add_argument("--b3p3", type=Path, required=True)
    p.add_argument("--ft-p0-640", type=Path, required=True)
    p.add_argument("--ft-p1-640", type=Path, required=True)
    p.add_argument("--ft-p0-896", type=Path, required=True)
    p.add_argument("--current-best", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    dfs = [
        load_probs(args.team, "team"),
        load_probs(args.perm8b, "perm"),
        load_probs(args.b3p3, "b3"),
        load_probs(args.ft_p0_640, "p0"),
        load_probs(args.ft_p1_640, "p1"),
        load_probs(args.ft_p0_896, "p896"),
    ]
    m = dfs[0]
    for df in dfs[1:]:
        m = m.merge(df, on="id", validate="one_to_one")
    if len(m) != 6714:
        raise ValueError(f"Expected 6714 rows, got {len(m)}")

    m["pred_p0"] = argmax_choice(m, "p0")
    m["pred_p1"] = argmax_choice(m, "p1")
    m["pred_p896"] = argmax_choice(m, "p896")
    m["override"] = (m["pred_p1"] == m["pred_p896"]) & (m["pred_p1"] != m["pred_p0"])

    for c in CHOICES:
        m[f"p_{c}_ftcons"] = m[f"p_{c}_p0"]
        mask = m["override"]
        m.loc[mask, f"p_{c}_ftcons"] = (
            0.5*m.loc[mask, f"p_{c}_p1"] + 0.5*m.loc[mask, f"p_{c}_p896"]
        )

    weights = {"team":0.7220, "perm":0.1805, "b3":0.0475, "ft":0.0500}
    score_cols = []
    for c in CHOICES:
        col = f"score_{c}"
        m[col] = (
            weights["team"]*np.log(m[f"p_{c}_team"].to_numpy()+1e-12)
            + weights["perm"]*np.log(m[f"p_{c}_perm"].to_numpy()+1e-12)
            + weights["b3"]*np.log(m[f"p_{c}_b3"].to_numpy()+1e-12)
            + weights["ft"]*np.log(m[f"p_{c}_ftcons"].to_numpy()+1e-12)
        )
        score_cols.append(col)

    labels = np.array(list(CHOICES))
    pred = labels[m[score_cols].to_numpy().argmax(axis=1)]

    current = pd.read_csv(args.current_best, encoding="utf-8-sig")
    cur_map = dict(zip(current["id"], current["answer"]))
    cur = m["id"].map(cur_map)
    if cur.isna().any():
        raise ValueError("Current-best ID mismatch")
    changed_mask = cur.to_numpy() != pred

    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    pred_map = dict(zip(m["id"], pred))
    sample["answer"] = sample["id"].map(pred_map)
    if sample["answer"].isna().any():
        raise ValueError("Sample submission ID mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sample.to_csv(args.output_dir/"submission.csv", index=False, encoding="utf-8-sig")

    m[["id","pred_p0","pred_p1","pred_p896","override"]].to_csv(
        args.output_dir/"crossview_decisions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "id":m.loc[changed_mask,"id"],
        "current_best_answer":cur[changed_mask].to_numpy(),
        "new_answer":pred[changed_mask],
        "pred_p0":m.loc[changed_mask,"pred_p0"],
        "pred_p1":m.loc[changed_mask,"pred_p1"],
        "pred_p896":m.loc[changed_mask,"pred_p896"],
        "override":m.loc[changed_mask,"override"],
    }).to_csv(args.output_dir/"changed_vs_current_best.csv", index=False, encoding="utf-8-sig")

    print("rows:", len(m))
    print("consensus overrides in FT slot:", int(m["override"].sum()))
    print("changed vs current best:", int(changed_mask.sum()))
    print("changed rate:", float(changed_mask.mean()))
    print("new distribution:", pd.Series(pred).value_counts().sort_index().to_dict())
    print("submission:", args.output_dir/"submission.csv")


if __name__ == "__main__":
    main()
