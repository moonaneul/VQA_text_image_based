from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def normalize_question(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\d+(?:[.,]\d+)*", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


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


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def union_by_key(dsu: DSU, keys: list[str]) -> None:
    first: dict[str, int] = {}
    for i, key in enumerate(keys):
        if key in first:
            dsu.union(i, first[key])
        else:
            first[key] = i


def distribution(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "answers": frame["answer"].astype(str).value_counts().sort_index().astype(int).to_dict(),
        "categories": frame["category"].value_counts().sort_index().astype(int).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a leakage-resistant QLoRA train/holdout split after excluding all used validation/audit IDs."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--random-val", type=Path, required=True)
    parser.add_argument("--grouped-val", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=Path("splits/qlora_train.csv"))
    parser.add_argument("--holdout-output", type=Path, default=Path("splits/qlora_holdout.csv"))
    parser.add_argument("--holdout-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    used_frames = [
        pd.read_csv(args.random_val, encoding="utf-8-sig"),
        pd.read_csv(args.grouped_val, encoding="utf-8-sig"),
        pd.read_csv(args.audit_csv, encoding="utf-8-sig"),
    ]

    for name, frame in [("train", train)] + [(f"used_{i}", f) for i, f in enumerate(used_frames)]:
        if "id" not in frame.columns:
            raise ValueError(f"{name} must contain id")

    required = {"path", "question", "answer"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train missing columns: {sorted(missing)}")

    used_ids = set()
    for frame in used_frames:
        used_ids.update(frame["id"].astype(str))

    pool = train[~train["id"].astype(str).isin(used_ids)].copy().reset_index(drop=True)
    pool["category"] = pool["question"].map(question_category)
    pool["_template"] = pool["question"].map(normalize_question)
    pool["_path_key"] = pool["path"].astype(str).str.replace("\\", "/", regex=False).str.lower()

    dsu = DSU(len(pool))
    union_by_key(dsu, pool["_template"].tolist())
    union_by_key(dsu, pool["_path_key"].tolist())

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pool)):
        components[dsu.find(i)].append(i)

    groups = list(components.values())
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    target = min(args.holdout_rows, len(pool))
    holdout_indices: list[int] = []
    remaining_groups = groups.copy()

    # Greedy target-size selection by shuffled connected components.
    selected_groups = []
    current = 0
    for group in groups:
        before = abs(target - current)
        after = abs(target - (current + len(group)))
        if current < target and (after <= before or current == 0):
            selected_groups.append(group)
            holdout_indices.extend(group)
            current += len(group)

    holdout_set = set(holdout_indices)
    holdout = pool.iloc[sorted(holdout_set)].copy()
    qtrain = pool.iloc[[i for i in range(len(pool)) if i not in holdout_set]].copy()

    template_overlap = set(qtrain["_template"]) & set(holdout["_template"])
    path_overlap = set(qtrain["_path_key"]) & set(holdout["_path_key"])
    if template_overlap or path_overlap:
        raise RuntimeError(
            f"Leakage detected: templates={len(template_overlap)}, paths={len(path_overlap)}"
        )

    drop_cols = ["category", "_template", "_path_key"]
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
    qtrain.drop(columns=drop_cols).to_csv(args.train_output, index=False, encoding="utf-8-sig")
    holdout.drop(columns=drop_cols).to_csv(args.holdout_output, index=False, encoding="utf-8-sig")

    manifest = {
        "original_train_rows": int(len(train)),
        "excluded_unique_ids": int(len(used_ids)),
        "safe_pool_rows": int(len(pool)),
        "connected_components": int(len(groups)),
        "requested_holdout_rows": int(args.holdout_rows),
        "actual_holdout_rows": int(len(holdout)),
        "qlora_train_rows": int(len(qtrain)),
        "template_overlap_train_holdout": int(len(template_overlap)),
        "path_overlap_train_holdout": int(len(path_overlap)),
        "train_distribution": distribution(qtrain),
        "holdout_distribution": distribution(holdout),
        "seed": args.seed,
        "train_output": str(args.train_output),
        "holdout_output": str(args.holdout_output),
    }

    manifest_path = args.holdout_output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
