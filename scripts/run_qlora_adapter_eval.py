from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass

from run_vlm_baseline import (
    CHOICES,
    PROMPTS,
    RESOLUTION_PRESETS,
    build_prompt,
    compute_metrics,
    extract_choice,
    model_input_device,
    question_category,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PEFT LoRA adapter on VQA data using the same generation path as B0."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/baseline"))
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="direct")
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS), default="standard")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--save-choice-probs",
        action="store_true",
        help="Save id,p_a,p_b,p_c,p_d using first-step logits normalized over choices a-d.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.csv, encoding="utf-8-sig")
    frame["category"] = frame["question"].map(question_category)

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
        **RESOLUTION_PRESETS[args.resolution],
    )

    base = ModelClass.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(
        base,
        args.adapter_path,
        is_trainable=False,
    )
    model.eval()

    choice_token_ids = {}
    if args.save_choice_probs:
        for choice in CHOICES:
            token_ids = processor.tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(
                    f"Choice {choice!r} is not a single token: {token_ids}. "
                    "Choice-probability export assumes one token per answer choice."
                )
            choice_token_ids[choice] = token_ids[0]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = model_input_device(model)
    records = []
    probability_records = []
    started = time.time()

    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="qlora-eval"):
        row_series = pd.Series(row._asdict())
        image_path = args.data_dir / Path(str(row.path).replace("\\", "/"))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_started = time.perf_counter()

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(row_series, args.prompt)},
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
            if args.save_choice_probs:
                generated_output = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                generated = generated_output.sequences

                first_step_scores = generated_output.scores[0][0]
                choice_logits = torch.stack([
                    first_step_scores[choice_token_ids[choice]]
                    for choice in CHOICES
                ]).float()
                choice_probs = torch.softmax(choice_logits, dim=0).detach().cpu().tolist()
                probability_records.append({
                    "id": row.id,
                    **{
                        f"p_{choice}": float(prob)
                        for choice, prob in zip(CHOICES, choice_probs)
                    },
                })
            else:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                )

        prompt_length = inputs["input_ids"].shape[1]
        text = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        prediction = extract_choice(text)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_seconds = time.perf_counter() - infer_started

        answer = getattr(row, "answer", None)
        is_labeled = answer in CHOICES
        records.append({
            "id": row.id,
            "prediction": prediction,
            "raw_output": text,
            "answer": answer,
            "correct": (prediction == answer) if is_labeled else None,
            "category": row.category,
            "latency_seconds": latency_seconds,
        })

    predictions = pd.DataFrame(records)
    predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    if args.save_choice_probs:
        probabilities = pd.DataFrame(probability_records)
        probabilities.to_csv(
            run_dir / "choice_probabilities.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics = compute_metrics(predictions)
    elapsed = time.time() - started
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        if torch.cuda.is_available()
        else None
    )

    result = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "input_csv": str(args.csv),
        "base_model": str(args.base_model),
        "adapter_path": str(args.adapter_path),
        "prompt": args.prompt,
        "resolution": args.resolution,
        "resolution_pixels": RESOLUTION_PRESETS[args.resolution],
        "quantization": "none",
        "decision_method": "generation",
        "choice_probabilities_saved": bool(args.save_choice_probs),
        "choice_probability_definition": (
            "first_generation_step_softmax_normalized_over_a_b_c_d"
            if args.save_choice_probs
            else None
        ),
        "seed": args.seed,
        "rows": int(len(predictions)),
        "elapsed_seconds": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "metrics": metrics,
    }
    (run_dir / "run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
