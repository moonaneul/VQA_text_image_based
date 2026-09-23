from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = "abcd"
PROB_COLS = [f"p_{c}" for c in CHOICES]
ROUTED_CATEGORIES = {"price", "spatial"}


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
    df = df.rename(columns={"score_a":"p_a","score_b":"p_b","score_c":"p_c","score_d":"p_d"})
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
    return df.rename(columns={c:f"{c}_{suffix}" for c in PROB_COLS})


def exact_mcnemar_p(rescue: int, regression: int) -> float:
    n = rescue + regression
    if n == 0:
        return 1.0
    k = min(rescue, regression)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def argmax_from(df: pd.DataFrame, suffix: str) -> pd.Series:
    cols = [f"p_{c}_{suffix}" for c in CHOICES]
    return df[cols].idxmax(axis=1).str.extract(r"p_([abcd])_", expand=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Validate/build a category-gated FT-8B profile: use 896 only for "
            "price/spatial, keep 640 elsewhere. Optionally build the final 5% "
            "slot ensemble candidate."
        )
    )
    p.add_argument("--data-csv", type=Path, required=True)
    p.add_argument("--ft640", type=Path, required=True)
    p.add_argument("--ft896", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--team", type=Path)
    p.add_argument("--perm8b", type=Path)
    p.add_argument("--b3p3", type=Path)
    p.add_argument("--current-best", type=Path)
    p.add_argument("--sample-submission", type=Path)
    args = p.parse_args()

    data = pd.read_csv(args.data_csv, encoding="utf-8-sig").copy()
    if "category" not in data.columns:
        data["category"] = data["question"].map(question_category)

    f640 = load_probs(args.ft640, "640")
    f896 = load_probs(args.ft896, "896")
    m = data.merge(f640, on="id", validate="one_to_one").merge(f896, on="id", validate="one_to_one")

    route = m["category"].isin(ROUTED_CATEGORIES)
    for c in CHOICES:
        m[f"p_{c}_routed"] = m[f"p_{c}_640"]
        m.loc[route, f"p_{c}_routed"] = m.loc[route, f"p_{c}_896"]

    m["pred_640"] = argmax_from(m, "640")
    m["pred_896"] = argmax_from(m, "896")
    routed_cols = [f"p_{c}_routed" for c in CHOICES]
    m["pred_routed"] = m[routed_cols].idxmax(axis=1).str.extract(r"p_([abcd])_routed", expand=False)

    summary = {
        "rows": int(len(m)),
        "routed_categories": sorted(ROUTED_CATEGORIES),
        "routed_rows": int(route.sum()),
        "ft_prediction_changes_vs_640": int((m["pred_routed"] != m["pred_640"]).sum()),
    }

    if "answer" in m.columns and m["answer"].isin(list(CHOICES)).any():
        labeled = m[m["answer"].isin(list(CHOICES))].copy()
        base_ok = labeled["pred_640"] == labeled["answer"]
        cand_ok = labeled["pred_routed"] == labeled["answer"]
        rescue = int((~base_ok & cand_ok).sum())
        regression = int((base_ok & ~cand_ok).sum())
        summary["validation"] = {
            "baseline_correct": int(base_ok.sum()),
            "candidate_correct": int(cand_ok.sum()),
            "baseline_accuracy": float(base_ok.mean()),
            "candidate_accuracy": float(cand_ok.mean()),
            "rescue": rescue,
            "regression": regression,
            "net": rescue - regression,
            "mcnemar_exact_p_value": exact_mcnemar_p(rescue, regression),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    m[["id","category","pred_640","pred_896","pred_routed"]].to_csv(
        args.output_dir/"ft_route_decisions.csv", index=False, encoding="utf-8-sig"
    )
    m[["id", *routed_cols]].rename(
        columns={f"p_{c}_routed":f"p_{c}" for c in CHOICES}
    ).to_csv(args.output_dir/"ft_routed_choice_probabilities.csv", index=False, encoding="utf-8-sig")

    build_final = all([
        args.team, args.perm8b, args.b3p3, args.current_best, args.sample_submission
    ])
    if build_final:
        team = load_probs(args.team, "team")
        perm = load_probs(args.perm8b, "perm")
        b3 = load_probs(args.b3p3, "b3")
        e = team.merge(perm, on="id", validate="one_to_one").merge(b3, on="id", validate="one_to_one")
        e = e.merge(
            m[["id", *routed_cols]],
            on="id",
            validate="one_to_one",
        )
        weights = {"team":0.7220,"perm":0.1805,"b3":0.0475,"ft":0.0500}
        score_cols = []
        for c in CHOICES:
            col = f"score_{c}"
            e[col] = (
                weights["team"]*np.log(e[f"p_{c}_team"].to_numpy()+1e-12)
                + weights["perm"]*np.log(e[f"p_{c}_perm"].to_numpy()+1e-12)
                + weights["b3"]*np.log(e[f"p_{c}_b3"].to_numpy()+1e-12)
                + weights["ft"]*np.log(e[f"p_{c}_routed"].to_numpy()+1e-12)
            )
            score_cols.append(col)
        labels = np.array(list(CHOICES))
        pred = labels[e[score_cols].to_numpy().argmax(axis=1)]

        cur = pd.read_csv(args.current_best, encoding="utf-8-sig")
        cur_map = dict(zip(cur["id"], cur["answer"]))
        cur_pred = e["id"].map(cur_map)
        changed = cur_pred.to_numpy() != pred

        sample = pd.read_csv(args.sample_submission, encoding="utf-8-sig")
        pred_map = dict(zip(e["id"], pred))
        sample["answer"] = sample["id"].map(pred_map)
        if sample["answer"].isna().any():
            raise ValueError("sample submission ID mismatch")
        sample.to_csv(args.output_dir/"submission.csv", index=False, encoding="utf-8-sig")

        changed_df = pd.DataFrame({
            "id": e.loc[changed,"id"],
            "current_best_answer": cur_pred[changed].to_numpy(),
            "new_answer": pred[changed],
        }).merge(m[["id","category","pred_640","pred_896","pred_routed"]], on="id", how="left")
        changed_df.to_csv(
            args.output_dir/"changed_vs_current_best.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary["final_ensemble"] = {
            "changed_vs_current_best": int(changed.sum()),
            "changed_rate": float(changed.mean()),
            "new_distribution": {
                str(k): int(v) for k,v in pd.Series(pred).value_counts().sort_index().items()
            },
        }

    (args.output_dir/"summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
