from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass


CHOICES = "abcd"
RESOLUTION_PRESETS = {
    "standard": {"min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28},
    "high": {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28},
}
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


def build_prompt(row: pd.Series) -> str:
    return (
        f"{DIRECT_PROMPT}\n\n"
        f"질문: {row['question']}\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}"
    )


def make_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    crop_w = max(1, int(round(w * 0.60)))
    crop_h = max(1, int(round(h * 0.60)))
    x1 = max(0, w - crop_w)
    y1 = max(0, h - crop_h)

    boxes = [
        ("top_left", (0, 0, crop_w, crop_h)),
        ("top_right", (x1, 0, w, crop_h)),
        ("bottom_left", (0, y1, crop_w, h)),
        ("bottom_right", (x1, y1, w, h)),
    ]
    views = [("full", image.copy())]
    views.extend((name, image.crop(box)) for name, box in boxes)
    return views


def load_model(model_path: Path, resolution: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
        **RESOLUTION_PRESETS[resolution],
    )
    model = ModelClass.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, processor


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choice_token_ids(processor) -> list[int]:
    tokenizer = processor.tokenizer
    ids = []
    for choice in CHOICES:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Choice {choice!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def score_view(model, processor, device, image: Image.Image, prompt: str, token_ids: list[int]):
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
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, token_ids].float()
        log_probs = torch.log_softmax(logits, dim=-1).cpu()

    return log_probs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B2 diagnostic: full image + overlapping 2x2 crops with choice-logit aggregation."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS), default="standard")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--run-id", default="B2D_scene_text_tiled_scoring")
    parser.add_argument("--output-root", type=Path, default=Path("output/baseline"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    frame["category"] = frame["question"].map(question_category)

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    model, processor = load_model(args.model_path, args.resolution)
    device = model_device(model)
    token_ids = choice_token_ids(processor)

    records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="tiled-scoring"):
        row_s = pd.Series(row._asdict())
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))
        prompt = build_prompt(row_s)

        with Image.open(image_path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            views = make_views(image)

        scores_by_view = {}
        for name, view in views:
            scores_by_view[name] = score_view(
                model, processor, device, view, prompt, token_ids
            )

        full_scores = scores_by_view["full"]
        crop_stack = torch.stack(
            [scores_by_view[name] for name in [
                "top_left", "top_right", "bottom_left", "bottom_right"
            ]]
        )
        max_crop_scores = crop_stack.max(dim=0).values

        # Primary aggregation is locked before evaluation:
        # global context from full image + strongest local crop evidence.
        combined = full_scores + max_crop_scores
        pred_idx = int(torch.argmax(combined).item())
        prediction = CHOICES[pred_idx]

        full_prediction = CHOICES[int(torch.argmax(full_scores).item())]
        best_crop_idx = int(torch.argmax(crop_stack[:, pred_idx]).item())
        best_crop = ["top_left", "top_right", "bottom_left", "bottom_right"][best_crop_idx]

        answer = getattr(row, "answer", None)
        records.append({
            "id": row.id,
            "prediction": prediction,
            "answer": answer,
            "correct": prediction == answer if answer in CHOICES else None,
            "category": row.category,
            "full_scoring_prediction": full_prediction,
            "best_crop_for_prediction": best_crop,
            "score_a": float(combined[0]),
            "score_b": float(combined[1]),
            "score_c": float(combined[2]),
            "score_d": float(combined[3]),
            "view_predictions": json.dumps({
                name: CHOICES[int(torch.argmax(scores).item())]
                for name, scores in scores_by_view.items()
            }, ensure_ascii=False),
        })

    pred = pd.DataFrame(records)
    pred.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {}
    if "answer" in pred.columns and pred["answer"].isin(list(CHOICES)).any():
        labeled = pred[pred["answer"].isin(list(CHOICES))].copy()
        correct = labeled["prediction"] == labeled["answer"]
        metrics = {
            "accuracy": float(correct.mean()),
            "correct": int(correct.sum()),
            "count": int(len(labeled)),
            "errors": int((~correct).sum()),
        }

    metadata = {
        "run_id": args.run_id,
        "rows": int(len(pred)),
        "resolution": args.resolution,
        "views": ["full", "top_left", "top_right", "bottom_left", "bottom_right"],
        "crop_fraction": 0.60,
        "aggregation": "full_logprob_plus_max_crop_logprob",
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
