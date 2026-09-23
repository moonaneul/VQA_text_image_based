from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]


def question_category(question: str) -> str:
    q = str(question).lower()
    if any(token in q for token in ["전화번호", "연락처"]):
        return "phone"
    if any(token in q for token in ["가격", "얼마", "금액", "원인가", "몇 원"]):
        return "price"
    if "메뉴" in q:
        return "menu"
    if any(token in q for token in ["적힌", "문구", "글자", "써 있", "상호명", "이름", "간판", "표지판"]):
        return "scene_text"
    if any(token in q for token in ["몇 개", "몇 명", "몇 마리", "개수", "수는"]):
        return "count"
    if any(token in q for token in ["무슨 색", "어떤 색", "색깔"]):
        return "color"
    if any(token in q for token in ["어디", "위치", "왼쪽", "오른쪽", "앞에", "뒤에"]):
        return "spatial"
    return "other"


def load_probs(path: Path, suffix: str) -> pd.DataFrame:
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
        raise ValueError(f"{path} has non-positive probability sums")
    df[PROB_COLS] = df[PROB_COLS].div(sums, axis=0)
    return df.rename(columns={c: f"{c}_{suffix}" for c in PROB_COLS})


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build the preregistered FT-8B price/spatial resolution-router test candidate. "
            "Use 896 FT probabilities only for price/spatial, 640 otherwise, then insert "
            "that routed FT profile into the existing 5% FT slot with all ensemble weights fixed."
        )
    )
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--team", type=Path, required=True)
    p.add_argument("--perm8b", type=Path, required=True)
    p.add_argument("--b3p3", type=Path, required=True)
    p.add_argument("--ft640", type=Path, required=True)
    p.add_argument("--ft896", type=Path, required=True)
    p.add_argument("--current-best", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    test = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    if not {"id", "question"}.issubset(test.columns):
        raise ValueError("test-csv must contain id and question")
    if test["id"].duplicated().any():
        raise ValueError("test-csv contains duplicate IDs")
    test = test[["id", "question"]].copy()
    test["category"] = test["question"].map(question_category)

    parts = [
        load_probs(args.team, "team"),
        load_probs(args.perm8b, "perm"),
        load_probs(args.b3p3, "b3"),
        load_probs(args.ft640, "ft640"),
        load_probs(args.ft896, "ft896"),
    ]
    m = test
    for df in parts:
        m = m.merge(df, on="id", validate="one_to_one")

    if len(m) != 6714:
        raise ValueError(f"Expected 6714 rows, got {len(m)}")

    route_mask = m["category"].isin(["price", "spatial"])
    for c in CHOICES:
        m[f"p_{c}_ftrouter"] = m[f"p_{c}_ft640"]
        m.loc[route_mask, f"p_{c}_ftrouter"] = m.loc[route_mask, f"p_{c}_ft896"]

    weights = {"team":0.7220, "perm":0.1805, "b3":0.0475, "ft":0.0500}
    score_cols = []
    for c in CHOICES:
        col = f"score_{c}"
        m[col] = (
            weights["team"] * np.log(m[f"p_{c}_team"].to_numpy() + 1e-12)
            + weights["perm"] * np.log(m[f"p_{c}_perm"].to_numpy() + 1e-12)
            + weights["b3"] * np.log(m[f"p_{c}_b3"].to_numpy() + 1e-12)
            + weights["ft"] * np.log(m[f"p_{c}_ftrouter"].to_numpy() + 1e-12)
        )
        score_cols.append(col)

    labels = np.array(list(CHOICES))
    pred = labels[m[score_cols].to_numpy().argmax(axis=1)]

    current = pd.read_csv(args.current_best, encoding="utf-8-sig")
    if not {"id", "answer"}.issubset(current.columns):
        raise ValueError("current-best must contain id and answer")
    cur_map = dict(zip(current["id"], current["answer"]))
    cur = m["id"].map(cur_map)
    if cur.isna().any():
        raise ValueError("current-best ID mismatch")
    changed_mask = cur.to_numpy() != pred

    sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
    pred_map = dict(zip(m["id"], pred))
    sample["answer"] = sample["id"].map(pred_map)
    if sample["answer"].isna().any():
        raise ValueError("sample submission ID mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)

    submission_path = args.output_dir / "submission.csv"
    sample.to_csv(submission_path, index=False, encoding="utf-8-sig")

    routed_probs = m[["id", *[f"p_{c}_ftrouter" for c in CHOICES]]].rename(
        columns={f"p_{c}_ftrouter": f"p_{c}" for c in CHOICES}
    )
    routed_probs.to_csv(
        args.output_dir / "ft8b_price_spatial_router_probs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changed = pd.DataFrame({
        "id": m.loc[changed_mask, "id"],
        "category": m.loc[changed_mask, "category"],
        "current_best_answer": cur[changed_mask].to_numpy(),
        "new_answer": pred[changed_mask],
    })
    changed.to_csv(
        args.output_dir / "changed_vs_current_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("rows:", len(m))
    print("routed test rows:", int(route_mask.sum()))
    print("route counts:", m.loc[route_mask, "category"].value_counts().sort_index().to_dict())
    print("changed vs current best:", int(changed_mask.sum()))
    print("changed rate:", float(changed_mask.mean()))
    print("changed by category:", changed["category"].value_counts().sort_index().to_dict())
    print("new distribution:", pd.Series(pred).value_counts().sort_index().to_dict())
    print("submission:", submission_path)


if __name__ == "__main__":
    main()
