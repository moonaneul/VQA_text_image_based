from __future__ import annotations

import argparse
import json
import math
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
VERIFY_LABELS = ("a", "b")  # a = candidate is correct, b = candidate is incorrect

VERIFY_PROMPT = (
    "이미지와 질문을 보고 아래 후보 답변이 실제 정답인지 판단하세요. "
    "후보 답변이 정답이면 a, 정답이 아니면 b로 판단하세요. "
    "최종 답은 a 또는 b 중 하나의 소문자 한 글자만 출력하세요."
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


def build_verify_prompt(question: str, candidate: str) -> str:
    return (
        f"{VERIFY_PROMPT}\n\n"
        f"질문: {question}\n"
        f"후보 답변: {candidate}\n\n"
        "판단 기준:\n"
        "- a: 후보 답변이 이미지와 질문에 비추어 정답임\n"
        "- b: 후보 답변이 정답이 아님"
    )


def exact_mcnemar_p_value(rescue: int, regression: int) -> float:
    n = rescue + regression
    if n == 0:
        return 1.0
    k = min(rescue, regression)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL option-verification scoring. Each of the four answer texts is "
            "verified independently as correct/incorrect; the choice with the highest "
            "binary correctness margin is selected."
        )
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
    args = parser.parse_args()

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

    verify_token_ids = {}
    for label in VERIFY_LABELS:
        ids = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Verification label {label!r} is not a single token: {ids}")
        verify_token_ids[label] = ids[0]

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

    yes_id = torch.tensor([verify_token_ids["a"]], dtype=torch.long, device=device)
    no_id = torch.tensor([verify_token_ids["b"]], dtype=torch.long, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = []
    prob_records = []
    detail_records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="option-verify"):
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        candidate_texts = {
            choice: str(getattr(row, choice))
            for choice in CHOICES
        }

        margins = {}
        yes_probs = {}

        for choice in CHOICES:
            prompt = build_verify_prompt(str(row.question), candidate_texts[choice])
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            chat = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[chat],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
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
                next_logits = outputs.logits[0, last_pos].float()
                yes_logit = next_logits.index_select(0, yes_id)[0]
                no_logit = next_logits.index_select(0, no_id)[0]
                margin = float((yes_logit - no_logit).detach().cpu().item())
                yes_prob = float(torch.sigmoid(yes_logit - no_logit).detach().cpu().item())

            margins[choice] = margin
            yes_probs[choice] = yes_prob

        # Margin and yes-probability induce the same ranking because sigmoid is monotonic.
        prediction = max(CHOICES, key=lambda c: margins[c])

        # Export a normalized four-choice distribution for downstream diagnostics.
        margin_tensor = torch.tensor([margins[c] for c in CHOICES], dtype=torch.float32)
        choice_probs = torch.softmax(margin_tensor, dim=0).tolist()

        answer = getattr(row, "answer", None)
        is_labeled = isinstance(answer, str) and answer in CHOICES

        records.append({
            "id": row.id,
            "prediction": prediction,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled else None,
            "category": row.category,
        })
        prob_records.append({
            "id": row.id,
            "p_a": float(choice_probs[0]),
            "p_b": float(choice_probs[1]),
            "p_c": float(choice_probs[2]),
            "p_d": float(choice_probs[3]),
        })
        detail_records.append({
            "id": row.id,
            **{f"margin_{c}": margins[c] for c in CHOICES},
            **{f"yesprob_{c}": yes_probs[c] for c in CHOICES},
            "prediction": prediction,
            "answer": answer,
            "category": row.category,
        })

    pred_df = pd.DataFrame(records)
    prob_df = pd.DataFrame(prob_records)
    detail_df = pd.DataFrame(detail_records)

    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    prob_df.to_csv(run_dir / "choice_probabilities.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(run_dir / "verification_scores.csv", index=False, encoding="utf-8-sig")

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
                    "correct": int(g["correct"].sum()),
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
        "decision_method": "four_independent_binary_option_verifications",
        "verification_labels": {"a": "candidate_correct", "b": "candidate_incorrect"},
        "choice_score": "logit(a)-logit(b)",
        "exported_choice_probabilities": "softmax_over_four_binary_margins",
        "rows": int(len(pred_df)),
        "forward_passes_per_row": 4,
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
