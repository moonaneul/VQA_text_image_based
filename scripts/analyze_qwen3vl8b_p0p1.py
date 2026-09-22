from __future__ import annotations

import argparse
import json
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
        raise ValueError(f"{path} has non-positive probability sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)
    return df.rename(columns={c: f"{c}_{suffix}" for c in PROB_COLS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average P0 and remapped P1 Qwen3-VL choice probabilities."
    )
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    p0 = load_probs(args.p0, "p0")
    p1 = load_probs(args.p1, "p1")
    eval_df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")

    required_eval = {"id", "answer", "question"}
    missing = required_eval - set(eval_df.columns)
    if missing:
        raise ValueError(f"{args.eval_csv} missing columns: {sorted(missing)}")
    if eval_df["id"].duplicated().any():
        raise ValueError(f"{args.eval_csv} contains duplicate IDs")

    m = eval_df.merge(p0, on="id", validate="one_to_one").merge(
        p1, on="id", validate="one_to_one"
    )
    if len(m) != len(eval_df):
        raise ValueError(
            f"ID join mismatch: eval={len(eval_df)} joined={len(m)}"
        )

    for c in CHOICES:
        m[f"p_{c}"] = 0.5 * (m[f"p_{c}_p0"] + m[f"p_{c}_p1"])

    m["pred_p0"] = m[[f"p_{c}_p0" for c in CHOICES]].idxmax(axis=1).str[2]
    m["pred_p1"] = m[[f"p_{c}_p1" for c in CHOICES]].idxmax(axis=1).str[2]
    m["pred_p0p1"] = m[PROB_COLS].idxmax(axis=1).str[2]

    m["p0_correct"] = m["pred_p0"] == m["answer"]
    m["p0p1_correct"] = m["pred_p0p1"] == m["answer"]

    rescue = int((~m["p0_correct"] & m["p0p1_correct"]).sum())
    regression = int((m["p0_correct"] & ~m["p0p1_correct"]).sum())
    changed = int((m["pred_p0"] != m["pred_p0p1"]).sum())
    p0_correct = int(m["p0_correct"].sum())
    p0p1_correct = int(m["p0p1_correct"].sum())

    args.output_dir.mkdir(parents=True, exist_ok=False)

    m[["id", *PROB_COLS]].to_csv(
        args.output_dir / "choice_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    m[
        [
            "id",
            "answer",
            "pred_p0",
            "pred_p1",
            "pred_p0p1",
            "p0_correct",
            "p0p1_correct",
        ]
    ].to_csv(
        args.output_dir / "paired_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "rows": int(len(m)),
        "p0_correct": p0_correct,
        "p0_accuracy": p0_correct / len(m),
        "p0p1_correct": p0p1_correct,
        "p0p1_accuracy": p0p1_correct / len(m),
        "rescue": rescue,
        "regression": regression,
        "net": rescue - regression,
        "changed_predictions": changed,
        "changed_rate": changed / len(m),
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
