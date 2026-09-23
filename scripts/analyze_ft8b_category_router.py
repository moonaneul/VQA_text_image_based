from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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


def exact_mcnemar(rescue: int, regression: int) -> float:
    n = rescue + regression
    if n == 0:
        return 1.0
    k = min(rescue, regression)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Validate a category router for FT-8B: use 896 only for price/spatial, "
            "otherwise keep 640."
        )
    )
    p.add_argument("--p0-640", type=Path, required=True)
    p.add_argument("--p0-896", type=Path, required=True)
    p.add_argument("--eval-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    p640 = load_probs(args.p0_640, "p640")
    p896 = load_probs(args.p0_896, "p896")
    eval_df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")
    required_eval = {"id", "answer", "question"}
    missing = required_eval - set(eval_df.columns)
    if missing:
        raise ValueError(f"{args.eval_csv} missing columns: {sorted(missing)}")
    eval_df["category"] = eval_df["question"].map(question_category)

    m = eval_df.merge(p640, on="id", validate="one_to_one").merge(
        p896, on="id", validate="one_to_one"
    )

    route_mask = m["category"].isin(["price", "spatial"])
    for c in CHOICES:
        m[f"p_{c}"] = m[f"p_{c}_p640"]
        m.loc[route_mask, f"p_{c}"] = m.loc[route_mask, f"p_{c}_p896"]

    m["pred_640"] = m[[f"p_{c}_p640" for c in CHOICES]].idxmax(axis=1).str.extract(r"p_([abcd])_", expand=False)
    m["pred_router"] = m[PROB_COLS].idxmax(axis=1).str[-1]
    m["correct_640"] = m["pred_640"] == m["answer"]
    m["correct_router"] = m["pred_router"] == m["answer"]

    rescue = int((~m["correct_640"] & m["correct_router"]).sum())
    regression = int((m["correct_640"] & ~m["correct_router"]).sum())
    changed = int((m["pred_640"] != m["pred_router"]).sum())
    base_correct = int(m["correct_640"].sum())
    router_correct = int(m["correct_router"].sum())

    result = {
        "rows": int(len(m)),
        "route_categories": ["price", "spatial"],
        "routed_rows": int(route_mask.sum()),
        "base_correct": base_correct,
        "base_accuracy": base_correct / len(m),
        "router_correct": router_correct,
        "router_accuracy": router_correct / len(m),
        "rescue": rescue,
        "regression": regression,
        "net": rescue - regression,
        "changed_predictions": changed,
        "changed_rate": changed / len(m),
        "mcnemar_exact_p_value": exact_mcnemar(rescue, regression),
        "by_category": {},
    }

    for cat, g in m.groupby("category"):
        result["by_category"][str(cat)] = {
            "count": int(len(g)),
            "base_correct": int(g["correct_640"].sum()),
            "router_correct": int(g["correct_router"].sum()),
            "net": int(g["correct_router"].sum() - g["correct_640"].sum()),
            "changed": int((g["pred_640"] != g["pred_router"]).sum()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    m[["id", *PROB_COLS]].to_csv(
        args.output_dir / "choice_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    m[m["pred_640"] != m["pred_router"]][
        ["id", "category", "answer", "pred_640", "pred_router", "correct_640", "correct_router"]
    ].to_csv(
        args.output_dir / "changed_vs_640.csv",
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
