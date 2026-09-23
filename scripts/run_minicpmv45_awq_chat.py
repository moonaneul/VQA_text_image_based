from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


CHOICES = "abcd"
DIRECT_PROMPT = (
    "이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요. "
    "반드시 a, b, c, d 중 하나의 소문자 한 글자만 답하세요."
)


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


def build_prompt(row) -> str:
    return (
        f"{DIRECT_PROMPT}\n\n"
        f"질문: {row.question}\n"
        f"(a) {row.a}\n"
        f"(b) {row.b}\n"
        f"(c) {row.c}\n"
        f"(d) {row.d}"
    )


def parse_choice(text: str) -> str | None:
    s = str(text).strip().lower()
    if s in CHOICES:
        return s

    # Accept a single standalone choice token in otherwise short output,
    # e.g. "정답은 b입니다."  Keep parsing deterministic and transparent.
    matches = re.findall(r"(?<![a-z])([abcd])(?![a-z])", s)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local MiniCPM-V-4.5-AWQ with its native chat inference."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/MiniCPM-V-4_5-AWQ"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/minicpmv"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(20260923)

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    frame["category"] = frame["question"].map(question_category)

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="minicpmv-chat"):
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        prompt = build_prompt(row)
        msgs = [{"role": "user", "content": [image, prompt]}]

        with torch.inference_mode():
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                enable_thinking=False,
                sampling=False,
                max_new_tokens=args.max_new_tokens,
            )

        prediction = parse_choice(response)
        answer = getattr(row, "answer", None)
        is_labeled = isinstance(answer, str) and answer in CHOICES

        records.append({
            "id": row.id,
            "prediction": prediction,
            "raw_response": str(response),
            "parsed": prediction in CHOICES,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled and prediction else False if is_labeled else None,
            "category": row.category,
        })

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    labeled = pred_df[pred_df["answer"].isin(list(CHOICES))].copy()
    metrics = {
        "rows": int(len(pred_df)),
        "parsed": int(pred_df["parsed"].sum()),
        "unparsed": int((~pred_df["parsed"]).sum()),
        "parse_rate": float(pred_df["parsed"].mean()) if len(pred_df) else 0.0,
    }

    if not labeled.empty:
        labeled["correct"] = labeled["prediction"] == labeled["answer"]
        metrics.update({
            "accuracy": float(labeled["correct"].mean()),
            "correct": int(labeled["correct"].sum()),
            "count": int(len(labeled)),
            "by_category": {
                str(k): {
                    "accuracy": float(g["correct"].mean()),
                    "correct": int(g["correct"].sum()),
                    "count": int(len(g)),
                }
                for k, g in labeled.groupby("category")
            },
        })

    result = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "csv": str(args.csv),
        "model_path": str(args.model_path),
        "decision_method": "native_chat_generation",
        "enable_thinking": False,
        "sampling": False,
        "max_new_tokens": args.max_new_tokens,
        "rows": int(len(pred_df)),
        "elapsed_seconds": time.time() - started,
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            if torch.cuda.is_available()
            else None
        ),
        "metrics": metrics,
    }
    (run_dir / "run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
