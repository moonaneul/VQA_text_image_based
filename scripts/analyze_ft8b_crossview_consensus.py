from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]


def load_probs(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", *PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    df = df[["id", *PROB_COLS]].copy()
    sums = df[PROB_COLS].sum(axis=1)
    if (sums <= 0).any():
        raise ValueError(f"{path} contains non-positive probability sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)
    return df.rename(columns={c: f"{c}_{suffix}" for c in PROB_COLS})


def exact_mcnemar_p_value(rescue: int, regression: int) -> float:
    n = rescue + regression
    if n == 0:
        return 1.0
    k = min(rescue, regression)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def argmax_choice(frame: pd.DataFrame, suffix: str) -> pd.Series:
    cols = [f"p_{c}_{suffix}" for c in CHOICES]
    return frame[cols].idxmax(axis=1).str.extract(r"p_([abcd])_", expand=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a conservative FT-8B cross-view consensus rule. "
            "Keep 640-P0 by default; override only when remapped 640-P1 and "
            "896-P0 agree with each other and disagree with 640-P0."
        )
    )
    parser.add_argument("--p0-640", type=Path, required=True)
    parser.add_argument("--p1-640", type=Path, required=True)
    parser.add_argument("--p0-896", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    p0 = load_probs(args.p0_640, "p0")
    p1 = load_probs(args.p1_640, "p1")
    p896 = load_probs(args.p0_896, "p896")
    eval_df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")

    required_eval = {"id", "answer"}
    missing = required_eval - set(eval_df.columns)
    if missing:
        raise ValueError(f"{args.eval_csv} missing columns: {sorted(missing)}")
    if eval_df["id"].duplicated().any():
        raise ValueError(f"{args.eval_csv} contains duplicate IDs")

    m = eval_df.merge(p0, on="id", validate="one_to_one")
    m = m.merge(p1, on="id", validate="one_to_one")
    m = m.merge(p896, on="id", validate="one_to_one")

    if len(m) != len(eval_df):
        raise ValueError(f"ID join mismatch: eval={len(eval_df)} joined={len(m)}")

    m["pred_p0"] = argmax_choice(m, "p0")
    m["pred_p1"] = argmax_choice(m, "p1")
    m["pred_p896"] = argmax_choice(m, "p896")

    # Fixed rule: two independent alternative views must agree before overriding P0.
    m["override"] = (
        (m["pred_p1"] == m["pred_p896"])
        & (m["pred_p1"] != m["pred_p0"])
    )

    # Candidate probabilities are P0 by default. On consensus overrides, use the
    # arithmetic mean of the two agreeing alternative views.
    for c in CHOICES:
        m[f"p_{c}_candidate"] = m[f"p_{c}_p0"]
        mask = m["override"]
        m.loc[mask, f"p_{c}_candidate"] = (
            0.5 * m.loc[mask, f"p_{c}_p1"]
            + 0.5 * m.loc[mask, f"p_{c}_p896"]
        )

    cand_cols = [f"p_{c}_candidate" for c in CHOICES]
    m["pred_candidate"] = m[cand_cols].idxmax(axis=1).str.extract(
        r"p_([abcd])_candidate", expand=False
    )

    m["p0_correct"] = m["pred_p0"] == m["answer"]
    m["candidate_correct"] = m["pred_candidate"] == m["answer"]

    rescue_mask = ~m["p0_correct"] & m["candidate_correct"]
    regression_mask = m["p0_correct"] & ~m["candidate_correct"]
    changed_mask = m["pred_p0"] != m["pred_candidate"]

    rescue = int(rescue_mask.sum())
    regression = int(regression_mask.sum())
    changed = int(changed_mask.sum())
    p0_correct = int(m["p0_correct"].sum())
    candidate_correct = int(m["candidate_correct"].sum())
    override_count = int(m["override"].sum())

    result = {
        "rows": int(len(m)),
        "rule": "keep_p0_except_when_p1_and_p896_agree_against_p0",
        "p0_correct": p0_correct,
        "p0_accuracy": p0_correct / len(m),
        "candidate_correct": candidate_correct,
        "candidate_accuracy": candidate_correct / len(m),
        "rescue": rescue,
        "regression": regression,
        "net": rescue - regression,
        "override_count": override_count,
        "changed_predictions": changed,
        "changed_rate": changed / len(m),
        "mcnemar_exact_p_value": exact_mcnemar_p_value(rescue, regression),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)

    out_probs = m[["id", *cand_cols]].rename(
        columns={f"p_{c}_candidate": f"p_{c}" for c in CHOICES}
    )
    out_probs.to_csv(
        args.output_dir / "choice_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    m.loc[
        changed_mask,
        [
            "id",
            "answer",
            "pred_p0",
            "pred_p1",
            "pred_p896",
            "pred_candidate",
            "p0_correct",
            "candidate_correct",
            "override",
        ],
    ].to_csv(
        args.output_dir / "changed_vs_p0.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
