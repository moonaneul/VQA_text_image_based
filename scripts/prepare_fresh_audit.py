from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fresh audit holdout excluding both existing validation splits."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--random-val", type=Path, required=True)
    parser.add_argument("--grouped-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--price-count", type=int, default=200)
    parser.add_argument("--phone-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    random_val = pd.read_csv(args.random_val, encoding="utf-8-sig")
    grouped_val = pd.read_csv(args.grouped_val, encoding="utf-8-sig")

    for name, frame in [
        ("train", train),
        ("random_val", random_val),
        ("grouped_val", grouped_val),
    ]:
        if "id" not in frame.columns:
            raise ValueError(f"{name} must contain id")

    if "question" not in train.columns:
        raise ValueError("train CSV must contain question")

    used_ids = set(random_val["id"].astype(str)) | set(grouped_val["id"].astype(str))
    pool = train[~train["id"].astype(str).isin(used_ids)].copy()
    pool["category"] = pool["question"].map(question_category)

    pieces = []
    requested = {"price": args.price_count, "phone": args.phone_count}
    availability = {}

    for category, count in requested.items():
        subset = pool[pool["category"] == category].copy()
        availability[category] = int(len(subset))
        if len(subset) < count:
            raise ValueError(
                f"Not enough untouched {category} rows: requested={count}, available={len(subset)}"
            )
        pieces.append(subset.sample(n=count, random_state=args.seed))

    audit = pd.concat(pieces, ignore_index=True)
    audit = audit.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.drop(columns=["category"]).to_csv(args.output, index=False, encoding="utf-8-sig")

    result = {
        "train_rows": int(len(train)),
        "excluded_unique_ids": int(len(used_ids)),
        "remaining_pool": int(len(pool)),
        "available_after_exclusion": availability,
        "audit_rows": int(len(audit)),
        "price_rows": int((audit["category"] == "price").sum()),
        "phone_rows": int((audit["category"] == "phone").sum()),
        "seed": args.seed,
        "output": str(args.output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
