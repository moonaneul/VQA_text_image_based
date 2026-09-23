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
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


CHOICES = "abcd"
KEY_MAPPING = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "vit_large_projector": "model.vit_large_projector",
}

DIRECT_PROMPT = (
    "이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요. "
    "필요한 추론은 수행하되 최종 답변은 반드시 마지막에 "
    "'FINAL: a', 'FINAL: b', 'FINAL: c', 'FINAL: d' 중 하나로 끝내세요."
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


def final_segment(text: str) -> str:
    s = str(text)
    if "</think>" in s:
        return s.rsplit("</think>", 1)[-1].strip()
    if "<|THINK_END|>" in s:
        return s.rsplit("<|THINK_END|>", 1)[-1].strip()
    return s.strip()


def parse_choice(text: str) -> str | None:
    s = final_segment(text).lower()

    matches = re.findall(r"final\s*[:：]\s*[\(\[]?([abcd])", s)
    if matches:
        return matches[-1]

    matches = re.findall(
        r"(?:정답|답|answer)\s*(?:은|는|:|：)?\s*[\(\[]?([abcd])",
        s,
    )
    if matches:
        return matches[-1]

    standalone = re.findall(r"(?<![a-z])([abcd])(?![a-z])", s)
    if standalone:
        return standalone[-1]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local Step3-VL-10B using experimental BNB NF4 4-bit inference."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Step3-VL-10B"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/step3vl"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(20260923)

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
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        key_mapping=KEY_MAPPING,
    ).eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="step3vl-chat"):
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        prompt = build_prompt(row)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_tokens = int(generated.shape[-1] - input_len)
        response = processor.decode(
            generated[0, input_len:],
            skip_special_tokens=True,
        ).strip()

        prediction = parse_choice(response)
        parsed = isinstance(prediction, str) and prediction in CHOICES
        answer = getattr(row, "answer", None)
        is_labeled = isinstance(answer, str) and answer in CHOICES

        records.append({
            "id": row.id,
            "prediction": prediction,
            "raw_response": response,
            "final_segment": final_segment(response),
            "parsed": parsed,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled and parsed else False if is_labeled else None,
            "category": row.category,
            "generated_tokens": generated_tokens,
            "hit_token_cap": generated_tokens >= args.max_new_tokens,
        })

        del inputs, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    labeled = pred_df[pred_df["answer"].isin(list(CHOICES))].copy()
    metrics = {
        "rows": int(len(pred_df)),
        "parsed": int(pred_df["parsed"].sum()),
        "unparsed": int((~pred_df["parsed"]).sum()),
        "parse_rate": float(pred_df["parsed"].mean()) if len(pred_df) else 0.0,
        "mean_generated_tokens": float(pred_df["generated_tokens"].mean()) if len(pred_df) else 0.0,
        "max_generated_tokens": int(pred_df["generated_tokens"].max()) if len(pred_df) else 0,
        "hit_token_cap": int(pred_df["hit_token_cap"].sum()) if len(pred_df) else 0,
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
        "decision_method": "thinking_chat_generation",
        "quantization": "experimental_bnb_nf4_4bit",
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
