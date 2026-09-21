from __future__ import annotations

import argparse
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
        description="Build a category-specific subset from any VQA CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    if "question" not in frame.columns:
        raise ValueError("input CSV must contain question")

    categories = frame["question"].map(question_category)
    subset = frame[categories == args.category].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{args.category}: {len(subset)} / {len(frame)} -> {args.output}")


if __name__ == "__main__":
    main()
