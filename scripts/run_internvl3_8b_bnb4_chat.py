from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


CHOICES = "abcd"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
        "<image>\n"
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
    matches = re.findall(r"(?<![a-z])([abcd])(?![a-z])", s)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def build_transform(input_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_ratio = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best = ratio
    return best


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: (x[0] * x[1], x[0], x[1]))

    target_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    resized = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
    processed: list[Image.Image] = []
    cols = target_width // image_size

    for idx in range(blocks):
        left = (idx % cols) * image_size
        upper = (idx // cols) * image_size
        box = (left, upper, left + image_size, upper + image_size)
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) > 1:
        processed.append(
            image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
        )
    return processed


def load_image(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    return torch.stack([transform(tile) for tile in tiles])


def get_vision_device(model) -> torch.device:
    try:
        return next(model.vision_model.parameters()).device
    except Exception:
        return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local InternVL3-8B with BNB 4-bit and native chat inference."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/InternVL3-8B"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/internvl3"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-num", type=int, default=12)
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
        use_fast=False,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=quant_config,
        dtype=torch.bfloat16,
    ).eval()

    vision_device = get_vision_device(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="internvl3-chat"):
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        pixel_values = load_image(image, input_size=448, max_num=args.max_num)
        pixel_values = pixel_values.to(device=vision_device, dtype=torch.bfloat16)
        prompt = build_prompt(row)

        with torch.inference_mode():
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )

        prediction = parse_choice(response)
        answer = getattr(row, "answer", None)
        is_labeled = isinstance(answer, str) and answer in CHOICES

        is_parsed = isinstance(prediction, str) and prediction in CHOICES
        records.append({
            "id": row.id,
            "prediction": prediction,
            "raw_response": str(response),
            "parsed": is_parsed,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled and is_parsed else False if is_labeled else None,
            "category": row.category,
            "image_tiles": int(pixel_values.shape[0]),
        })

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    labeled = pred_df[pred_df["answer"].isin(list(CHOICES))].copy()
    metrics = {
        "rows": int(len(pred_df)),
        "parsed": int(pred_df["parsed"].sum()),
        "unparsed": int((~pred_df["parsed"]).sum()),
        "parse_rate": float(pred_df["parsed"].mean()) if len(pred_df) else 0.0,
        "mean_image_tiles": float(pred_df["image_tiles"].mean()) if len(pred_df) else 0.0,
        "max_image_tiles": int(pred_df["image_tiles"].max()) if len(pred_df) else 0,
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
        "quantization": "bnb_nf4_4bit",
        "sampling": False,
        "max_new_tokens": args.max_new_tokens,
        "max_num": args.max_num,
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
