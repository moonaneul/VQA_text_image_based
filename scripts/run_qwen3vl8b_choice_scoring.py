from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass


CHOICES = "abcd"
DIRECT_PROMPT = (
    "이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요. "
    "최종 답은 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요."
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


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(row, choice_order: str = CHOICES) -> str:
    options = {choice: getattr(row, choice) for choice in CHOICES}
    lines = [
        f"{DIRECT_PROMPT}\n\n질문: {row.question}",
    ]
    for display_label, original_choice in zip(CHOICES, choice_order):
        lines.append(f"({display_label}) {options[original_choice]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL with four-choice next-token scoring."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("output/qwen3vl8b"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--choice-order",
        default=CHOICES,
        help=(
            "Original-choice order displayed as labels a/b/c/d. "
            "Use 'abcd' for P0 or 'bcda' for P1. "
            "Saved p_a..p_d are always remapped back to original-choice identity."
        ),
    )
    args = parser.parse_args()

    if len(args.choice_order) != 4 or set(args.choice_order) != set(CHOICES):
        raise ValueError(
            f"--choice-order must be a permutation of {CHOICES!r}; got {args.choice_order!r}"
        )

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    frame["category"] = frame["question"].map(question_category)

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    choice_token_ids = []
    for choice in CHOICES:
        ids = processor.tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Choice {choice!r} is not a single token: {ids}")
        choice_token_ids.append(ids[0])

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = ModelClass.from_pretrained(args.model_path, **model_kwargs)
    model = (
        PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
        if args.adapter_path
        else base
    )
    model.eval()
    device = model_device(model)
    choice_ids_tensor = torch.tensor(choice_token_ids, dtype=torch.long, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = []
    probs = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="qwen3-score"):
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(row, args.choice_order)},
            ],
        }]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], padding=False, return_tensors="pt")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)
            if "attention_mask" in inputs:
                last_pos = int(inputs["attention_mask"][0].sum().item()) - 1
            else:
                last_pos = int(inputs["input_ids"].shape[1]) - 1
            next_logits = outputs.logits[0, last_pos]
            choice_logits = next_logits.index_select(0, choice_ids_tensor).float()
            p_display = torch.softmax(choice_logits, dim=0).detach().cpu().tolist()

        # Display label a/b/c/d may point to a permuted original choice.
        # Remap probabilities back to original choice identity so downstream
        # artifacts always use p_a/p_b/p_c/p_d consistently.
        p = [0.0, 0.0, 0.0, 0.0]
        for display_idx, original_choice in enumerate(args.choice_order):
            p[CHOICES.index(original_choice)] = float(p_display[display_idx])

        pred_idx = max(range(4), key=lambda i: p[i])
        prediction = CHOICES[pred_idx]
        answer = getattr(row, "answer", None)
        is_labeled = isinstance(answer, str) and answer in CHOICES

        records.append({
            "id": row.id,
            "prediction": prediction,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled else None,
            "category": row.category,
        })
        probs.append({
            "id": row.id,
            "p_a": float(p[0]),
            "p_b": float(p[1]),
            "p_c": float(p[2]),
            "p_d": float(p[3]),
        })

    pred_df = pd.DataFrame(records)
    prob_df = pd.DataFrame(probs)
    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    prob_df.to_csv(run_dir / "choice_probabilities.csv", index=False, encoding="utf-8-sig")

    labeled = pred_df[pred_df["answer"].isin(list(CHOICES))].copy()
    metrics = {}
    if not labeled.empty:
        labeled["correct"] = labeled["prediction"] == labeled["answer"]
        metrics = {
            "accuracy": float(labeled["correct"].mean()),
            "correct": int(labeled["correct"].sum()),
            "count": int(len(labeled)),
            "by_category": {
                str(k): {
                    "accuracy": float(g["correct"].mean()),
                    "count": int(len(g)),
                }
                for k, g in labeled.groupby("category")
            },
        }

    result = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "csv": str(args.csv),
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "decision_method": "four_choice_next_token_scoring",
        "choice_order": args.choice_order,
        "probability_identity": "original_choice_after_remap",
        "rows": int(len(pred_df)),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "quantization": args.quantization,
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
