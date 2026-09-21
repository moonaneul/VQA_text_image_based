from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def normalize_question(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"\d+(?:[.,:/-]\d+)*", "<num>", text)
    text = re.sub(r"[\"'‘’“”]", "", text)
    return re.sub(r"\s+", " ", text)


def question_category(question: str) -> str:
    q = unicodedata.normalize("NFKC", str(question)).lower()
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


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_hashes(frame: pd.DataFrame, data_dir: Path, metadata_path: Path | None) -> dict[str, str]:
    if metadata_path and metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        train_metadata = metadata[metadata["split"] == "train"]
        return dict(zip(train_metadata["id"].astype(str), train_metadata["sha256"].astype(str)))
    return {
        str(row.id): sha256_file(data_dir / Path(str(row.path).replace("\\", "/")))
        for row in frame.itertuples(index=False)
    }


def random_stratified_indices(frame: pd.DataFrame, val_fraction: float, seed: int) -> set[int]:
    rng = random.Random(seed)
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        strata[f"{row['answer']}|{row['category']}"].append(index)
    selected: set[int] = set()
    for indices in strata.values():
        rng.shuffle(indices)
        count = max(1, round(len(indices) * val_fraction)) if len(indices) >= 2 else 0
        selected.update(indices[:count])
    return selected


def grouped_indices(frame: pd.DataFrame, image_hash_by_id: dict[str, str], val_fraction: float, seed: int) -> set[int]:
    union_find = UnionFind(len(frame))
    first_by_question: dict[str, int] = {}
    first_by_image: dict[str, int] = {}
    for index, row in frame.iterrows():
        question_key = row["question_template"]
        image_key = image_hash_by_id.get(str(row["id"]), "")
        if question_key in first_by_question:
            union_find.union(index, first_by_question[question_key])
        else:
            first_by_question[question_key] = index
        if image_key:
            if image_key in first_by_image:
                union_find.union(index, first_by_image[image_key])
            else:
                first_by_image[image_key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in frame.index:
        groups[union_find.find(index)].append(index)

    target_total = round(len(frame) * val_fraction)
    total_answers = Counter(frame["answer"])
    total_categories = Counter(frame["category"])
    target_answers = {key: value * val_fraction for key, value in total_answers.items()}
    target_categories = {key: value * val_fraction for key, value in total_categories.items()}
    group_items = list(groups.values())

    def score(indices: set[int]) -> float:
        subset = frame.loc[sorted(indices)]
        answers = Counter(subset["answer"])
        categories = Counter(subset["category"])
        answer_error = sum(
            (answers[key] - target) ** 2 / max(target, 1.0)
            for key, target in target_answers.items()
        )
        category_error = sum(
            (categories[key] - target) ** 2 / max(target, 1.0)
            for key, target in target_categories.items()
        )
        return answer_error + category_error

    best_indices: set[int] | None = None
    best_score = float("inf")
    # Repeated deterministic group shuffles are fast for this dataset and avoid
    # over-selecting the largest repeated OCR templates.
    for trial in range(256):
        rng = random.Random(seed + trial)
        order = group_items.copy()
        rng.shuffle(order)
        candidate: set[int] = set()
        for indices in order:
            if len(candidate) + len(indices) <= target_total:
                candidate.update(indices)
            if len(candidate) == target_total:
                break
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best_indices = candidate
            best_score = candidate_score
    if best_indices is None:
        raise RuntimeError("Could not construct grouped validation split")
    return best_indices


def leakage_summary(train: pd.DataFrame, val: pd.DataFrame, image_hash_by_id: dict[str, str]) -> dict:
    train_questions = set(train["question_template"])
    val_questions = set(val["question_template"])
    train_images = {image_hash_by_id.get(str(value), "") for value in train["id"]} - {""}
    val_images = {image_hash_by_id.get(str(value), "") for value in val["id"]} - {""}
    return {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "question_template_overlap": int(len(train_questions & val_questions)),
        "exact_image_overlap": int(len(train_images & val_images)),
        "train_answer_distribution": {str(k): int(v) for k, v in train["answer"].value_counts().sort_index().items()},
        "val_answer_distribution": {str(k): int(v) for k, v in val["answer"].value_counts().sort_index().items()},
        "train_category_distribution": {str(k): int(v) for k, v in train["category"].value_counts().items()},
        "val_category_distribution": {str(k): int(v) for k, v in val["category"].value_counts().items()},
    }


def save_split(frame: pd.DataFrame, val_indices: set[int], output_dir: Path, name: str) -> dict:
    val = frame.loc[sorted(val_indices)].copy()
    train = frame.drop(index=val_indices).copy()
    exported_columns = [column for column in frame.columns if column not in {"question_template", "category"}]
    train[exported_columns].to_csv(output_dir / f"train_{name}.csv", index=False, encoding="utf-8-sig")
    val[exported_columns].to_csv(output_dir / f"val_{name}.csv", index=False, encoding="utf-8-sig")
    return {"train": train, "val": val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic random and leakage-controlled VQA validation splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--metadata", type=Path, default=Path("reports/eda/image_metadata.csv"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")

    frame = pd.read_csv(args.data_dir / "train.csv", encoding="utf-8-sig").reset_index(drop=True)
    frame["question_template"] = frame["question"].map(normalize_question)
    frame["category"] = frame["question"].map(question_category)
    hashes = image_hashes(frame, args.data_dir, args.metadata if args.metadata.is_file() else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random_val = random_stratified_indices(frame, args.val_fraction, args.seed)
    grouped_val = grouped_indices(frame, hashes, args.val_fraction, args.seed)
    random_split = save_split(frame, random_val, args.output_dir, "random")
    grouped_split = save_split(frame, grouped_val, args.output_dir, "grouped")

    summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "random": leakage_summary(random_split["train"], random_split["val"], hashes),
        "grouped": leakage_summary(grouped_split["train"], grouped_split["val"], hashes),
        "group_definition": "Connected components sharing a normalized question template or exact image SHA-256.",
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
