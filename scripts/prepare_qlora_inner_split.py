from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an inner dev split for QLoRA tuning without touching the final holdout."
    )
    parser.add_argument("--qlora-train", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=Path("splits/qlora_inner_train.csv"))
    parser.add_argument("--dev-output", type=Path, default=Path("splits/qlora_inner_dev.csv"))
    parser.add_argument("--dev-rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    frame = pd.read_csv(args.qlora_train, encoding="utf-8-sig").reset_index(drop=True)
    required = {"id", "path", "question", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"qlora_train missing columns: {sorted(missing)}")

    frame["_template"] = frame["question"].map(normalize_question)
    frame["_path_key"] = frame["path"].astype(str).str.replace("\\", "/", regex=False).str.lower()

    dsu = DSU(len(frame))
    union_by_key(dsu, frame["_template"].tolist())
    union_by_key(dsu, frame["_path_key"].tolist())

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(frame)):
        components[dsu.find(i)].append(i)

    groups = list(components.values())
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    target = min(args.dev_rows, len(frame))
    dev_indices: list[int] = []
    current = 0
    for group in groups:
        before = abs(target - current)
        after = abs(target - (current + len(group)))
        if current < target and (after <= before or current == 0):
            dev_indices.extend(group)
            current += len(group)

    dev_set = set(dev_indices)
    dev = frame.iloc[sorted(dev_set)].copy()
    train = frame.iloc[[i for i in range(len(frame)) if i not in dev_set]].copy()

    template_overlap = set(train["_template"]) & set(dev["_template"])
    path_overlap = set(train["_path_key"]) & set(dev["_path_key"])
    if template_overlap or path_overlap:
        raise RuntimeError(
            f"Leakage detected: templates={len(template_overlap)}, paths={len(path_overlap)}"
        )

    drop = ["_template", "_path_key"]
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train.drop(columns=drop).to_csv(args.train_output, index=False, encoding="utf-8-sig")
    dev.drop(columns=drop).to_csv(args.dev_output, index=False, encoding="utf-8-sig")

    result = {
        "source_rows": int(len(frame)),
        "inner_train_rows": int(len(train)),
        "inner_dev_rows": int(len(dev)),
        "connected_components": int(len(groups)),
        "template_overlap_train_dev": int(len(template_overlap)),
        "path_overlap_train_dev": int(len(path_overlap)),
        "answer_distribution_train": train["answer"].astype(str).value_counts().sort_index().astype(int).to_dict(),
        "answer_distribution_dev": dev["answer"].astype(str).value_counts().sort_index().astype(int).to_dict(),
        "seed": args.seed,
        "train_output": str(args.train_output),
        "dev_output": str(args.dev_output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
