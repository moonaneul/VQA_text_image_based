from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass


CHOICES = "abcd"
RESOLUTION = {
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1024 * 28 * 28,
}
DIRECT_PROMPT = (
    "이미지와 질문, 네 개의 선택지를 보고 정답을 고르세요. "
    "최종 답은 a, b, c, d 중 하나의 소문자 한 글자만 출력하세요."
)


class VQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        return self.frame.iloc[index].to_dict()


def build_user_text(row: dict) -> str:
    return (
        f"{DIRECT_PROMPT}\n\n"
        f"질문: {row['question']}\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}"
    )


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_answer_only_batch(processor, row: dict, data_dir: Path, device: torch.device) -> dict:
    image_path = data_dir / Path(str(row["path"]).replace("\\", "/"))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    user_text = build_user_text(row)
    user_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    }]
    full_messages = user_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": str(row["answer"])}],
    }]

    prompt_chat = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_chat = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_inputs = processor(
        text=[prompt_chat],
        images=[image],
        padding=False,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_chat],
        images=[image],
        padding=False,
        return_tensors="pt",
    )

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    input_ids = full_inputs["input_ids"]
    answer = str(row["answer"]).strip().lower()
    if answer not in CHOICES:
        raise ValueError(f"Invalid answer: {answer!r}")

    answer_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    if len(answer_ids) != 1:
        raise ValueError(f"Answer {answer!r} is not one token: {answer_ids}")
    answer_id = answer_ids[0]

    suffix = input_ids[0, prompt_len:]
    matches = (suffix == answer_id).nonzero(as_tuple=False).flatten()
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one answer token after prompt for id={row['id']}; "
            f"answer={answer!r}, matches={len(matches)}, prompt_len={prompt_len}, "
            f"full_len={input_ids.shape[1]}"
        )

    answer_pos = prompt_len + int(matches.item())
    labels = torch.full_like(input_ids, -100)
    labels[0, answer_pos] = answer_id

    batch = dict(full_inputs)
    batch["labels"] = labels
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative answer-token-only QLoRA training for Qwen2.5-VL."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--save-every-steps", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
        **RESOLUTION,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = ModelClass.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_trainable = [parameter.detach().cpu().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    dataset = VQADataset(frame)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(round(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    device = model_device(model)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    micro_step = 0
    loss_sum = 0.0
    first_grad_norm = None
    first_grad_param_count = None
    started = time.time()

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for row in progress:
            batch = build_answer_only_batch(processor, row, args.data_dir, device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            raw_loss = float(outputs.loss.detach().cpu())
            if not math.isfinite(raw_loss):
                raise RuntimeError(f"Non-finite loss detected at micro_step={micro_step + 1}: {raw_loss}")
            loss_sum += raw_loss
            micro_step += 1

            if first_grad_norm is None:
                grad_sq = 0.0
                grad_count = 0
                for parameter in trainable:
                    if parameter.grad is not None:
                        grad = parameter.grad.detach().float()
                        grad_sq += float(torch.sum(grad * grad).cpu())
                        if torch.any(grad != 0):
                            grad_count += 1
                first_grad_norm = math.sqrt(grad_sq)
                first_grad_param_count = grad_count
                if first_grad_norm == 0.0 or first_grad_param_count == 0:
                    raise RuntimeError(
                        "No non-zero LoRA gradients detected on the first microbatch. "
                        "Stop before full training."
                    )

            should_step = (micro_step % args.grad_accum == 0) or (micro_step == len(loader) * args.epochs)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if args.save_every_steps and global_update % args.save_every_steps == 0:
                    checkpoint = args.output_dir / f"checkpoint-{global_update}"
                    model.save_pretrained(checkpoint)

            progress.set_postfix({
                "loss": f"{raw_loss:.4f}",
                "updates": global_update,
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    adapter_change_sq = 0.0
    for parameter, initial in zip(trainable, initial_trainable):
        delta = parameter.detach().cpu().float() - initial.float()
        adapter_change_sq += float(torch.sum(delta * delta))
    adapter_parameter_l2_change = math.sqrt(adapter_change_sq)
    if adapter_parameter_l2_change == 0.0:
        raise RuntimeError("LoRA parameters did not change during training.")

    metadata = {
        "train_csv": str(args.train_csv),
        "train_rows": int(len(frame)),
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "quantization": "4bit_nf4_double_quant",
        "compute_dtype": "bfloat16",
        "resolution": RESOLUTION,
        "loss_masking": "assistant_answer_token_only",
        "optimizer": "AdamW",
        "total_updates": global_update,
        "mean_microbatch_loss": loss_sum / max(1, micro_step),
        "first_grad_norm": first_grad_norm,
        "first_grad_param_count": first_grad_param_count,
        "adapter_parameter_l2_change": adapter_parameter_l2_change,
        "gradient_checkpointing": "non_reentrant",
        "elapsed_seconds": time.time() - started,
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            if torch.cuda.is_available()
            else None
        ),
        "seed": args.seed,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
