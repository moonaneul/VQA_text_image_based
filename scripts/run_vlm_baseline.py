from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:  # compatibility fallback for newer Transformers releases
    from transformers import AutoModelForImageTextToText as ModelClass


CHOICES = "abcd"
RESOLUTION_PRESETS = {
    "low": {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28},
    "standard": {"min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28},
    "high": {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28},
}
PROMPTS = {
    "direct": (
        "이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요. "
        "최종 답은 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요."
    ),
    "ocr_deliberate": (
        "이미지 속 글자와 숫자를 확대해 읽듯이 세밀하게 확인한 뒤 질문에 답하세요. "
        "간판, 메뉴, 가격, 전화번호, 날짜의 정확한 표기를 선택지와 비교하세요. "
        "최종 답은 설명 없이 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요."
    ),
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_prompt(row: pd.Series, prompt_name: str) -> str:
    return (
        f"{PROMPTS[prompt_name]}\n\n"
        f"질문: {row['question']}\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}"
    )


def extract_choice(generated_text: str) -> str | None:
    text = generated_text.strip().lower()
    if text in CHOICES:
        return text
    match = re.search(r"(?:^|[^a-z])\(?([abcd])\)?(?:[^a-z]|$)", text)
    return match.group(1) if match else None


def load_model(model_path: Path, resolution: str, quantization: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
        **RESOLUTION_PRESETS[resolution],
    )
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = ModelClass.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, processor


def model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_one(model, processor, image: Image.Image, prompt: str) -> tuple[str | None, str]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
    device = model_input_device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return extract_choice(text), text


def compute_metrics(predictions: pd.DataFrame) -> dict:
    valid = predictions[predictions["answer"].isin(list(CHOICES))].copy()
    if valid.empty:
        return {}
    valid["correct"] = valid["prediction"] == valid["answer"]
    by_category = valid.groupby("category")["correct"].agg(["mean", "count"])
    return {
        "accuracy": float(valid["correct"].mean()),
        "correct": int(valid["correct"].sum()),
        "count": int(len(valid)),
        "unparsed": int(predictions["prediction"].isna().sum()),
        "by_category": {
            str(category): {"accuracy": float(row["mean"]), "count": int(row["count"])}
            for category, row in by_category.iterrows()
        },
    }


def append_experiment(log_path: Path, record: dict) -> None:
    fields = [
        "run_id", "timestamp", "stage", "model", "prompt", "resolution", "quantization",
        "split", "seed", "rows", "overall_accuracy", "scene_text_accuracy", "phone_accuracy",
        "price_accuracy", "menu_accuracy", "count_accuracy", "color_accuracy",
        "spatial_accuracy", "other_accuracy", "lb_score", "notes", "artifacts",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.is_file() and log_path.stat().st_size > 0
    with log_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible local Qwen VLM zero-shot baseline.")
    parser.add_argument("--csv", type=Path, required=True, help="Validation CSV with answer, or test CSV without answer.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--output-root", type=Path, default=Path("output/baseline"))
    parser.add_argument("--experiment-log", type=Path, default=Path("experiments/experiments.csv"))
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="direct")
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS), default="standard")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()
    frame["category"] = frame["question"].map(question_category)
    mode = "validation" if "answer" in frame.columns else "inference"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}_{args.prompt}_{args.resolution}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    model, processor = load_model(args.model_path, args.resolution, args.quantization)
    records = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=mode):
        row_series = pd.Series(row._asdict())
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            prediction, raw_output = predict_one(model, processor, image, build_prompt(row_series, args.prompt))
        records.append({
            "id": row.id,
            "prediction": prediction,
            "raw_output": raw_output,
            "answer": getattr(row, "answer", None),
            "category": row.category,
        })

    predictions = pd.DataFrame(records)
    predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = compute_metrics(predictions)
    if mode == "inference":
        if predictions["prediction"].isna().any():
            raise RuntimeError("Submission was not created because at least one response could not be parsed. Inspect predictions.csv.")
        submission = pd.DataFrame({"id": predictions["id"], "answer": predictions["prediction"]})
        submission.to_csv(run_dir / "submission.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - started
    run_metadata = {
        "run_id": run_id,
        "mode": mode,
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_csv": str(args.csv),
        "input_csv_sha256": sha256_file(args.csv),
        "model_path": str(args.model_path),
        "prompt": args.prompt,
        "prompt_text": PROMPTS[args.prompt],
        "resolution": args.resolution,
        "resolution_pixels": RESOLUTION_PRESETS[args.resolution],
        "quantization": args.quantization,
        "seed": args.seed,
        "rows": len(frame),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(frame) / elapsed if elapsed else None,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    by_category = metrics.get("by_category", {})
    append_experiment(args.experiment_log, {
        "run_id": run_id,
        "timestamp": run_metadata["timestamp"],
        "stage": "zero-shot",
        "model": args.model_path.name,
        "prompt": args.prompt,
        "resolution": args.resolution,
        "quantization": args.quantization,
        "split": args.csv.name,
        "seed": args.seed,
        "rows": len(frame),
        "overall_accuracy": metrics.get("accuracy", ""),
        **{f"{category}_accuracy": by_category.get(category, {}).get("accuracy", "")
           for category in ["scene_text", "phone", "price", "menu", "count", "color", "spatial", "other"]},
        "artifacts": str(run_dir),
        "notes": f"smoke test: {args.max_samples} rows" if args.max_samples else "",
    })
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
