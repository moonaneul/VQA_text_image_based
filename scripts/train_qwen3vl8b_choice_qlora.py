from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup

try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass


CHOICES = "abcd"
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 640 * 28 * 28
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


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remap_choice_order(row: dict, rng: random.Random, shuffle_prob: float) -> tuple[dict, str, str]:
    original_answer = str(row["answer"]).strip().lower()
    if original_answer not in CHOICES:
        raise ValueError(f"Invalid answer: {original_answer!r}")

    pairs = [(label, str(row[label])) for label in CHOICES]
    applied = rng.random() < shuffle_prob
    if applied:
        rng.shuffle(pairs)

    displayed = {label: pairs[i][1] for i, label in enumerate(CHOICES)}
    new_answer = next(
        display_label
        for display_label, (original_label, _) in zip(CHOICES, pairs)
        if original_label == original_answer
    )
    permutation = "".join(original_label for original_label, _ in pairs)
    return displayed, new_answer, permutation


def build_prompt(row: dict, displayed: dict[str, str]) -> str:
    return (
        f"{DIRECT_PROMPT}\n\n"
        f"질문: {row['question']}\n"
        f"(a) {displayed['a']}\n"
        f"(b) {displayed['b']}\n"
        f"(c) {displayed['c']}\n"
        f"(d) {displayed['d']}"
    )


def build_batch(processor, row: dict, data_dir: Path, device: torch.device, rng: random.Random, shuffle_prob: float):
    image_path = data_dir / Path(str(row["path"]).replace("\\", "/"))
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    displayed, answer, permutation = remap_choice_order(row, rng, shuffle_prob)
    prompt = build_prompt(row, displayed)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=False, return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    return inputs, answer, permutation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B QLoRA with four-choice classification loss."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--choice-shuffle-prob", type=float, default=0.50)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    if not 0.0 <= args.choice_shuffle_prob <= 1.0:
        raise ValueError("--choice-shuffle-prob must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("This training recipe expects CUDA.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    frame = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    required = {"id", "path", "question", "a", "b", "c", "d", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"train CSV missing columns: {sorted(missing)}")
    if args.max_samples:
        frame = frame.head(args.max_samples).copy()

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

    trainable = [p for p in model.parameters() if p.requires_grad]
    initial_trainable = [p.detach().cpu().clone() for p in trainable]
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

    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = model_device(model)
    optimizer.zero_grad(set_to_none=True)

    micro_step = 0
    global_update = 0
    loss_sum = 0.0
    first_grad_norm = None
    first_grad_param_count = None
    permutation_counts: dict[str, int] = {}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    choice_ids_tensor = torch.tensor(choice_token_ids, dtype=torch.long, device=device)

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for row in progress:
            inputs, answer, permutation = build_batch(
                processor, row, args.data_dir, device, rng, args.choice_shuffle_prob
            )
            permutation_counts[permutation] = permutation_counts.get(permutation, 0) + 1

            outputs = model(**inputs)
            if "attention_mask" in inputs:
                last_pos = int(inputs["attention_mask"][0].sum().item()) - 1
            else:
                last_pos = int(inputs["input_ids"].shape[1]) - 1

            next_logits = outputs.logits[0, last_pos]
            choice_logits = next_logits.index_select(0, choice_ids_tensor).float().unsqueeze(0)
            target = torch.tensor([CHOICES.index(answer)], dtype=torch.long, device=device)
            raw_loss_tensor = F.cross_entropy(choice_logits, target)
            raw_loss = float(raw_loss_tensor.detach().cpu())
            if not math.isfinite(raw_loss):
                raise RuntimeError(f"Non-finite loss at micro_step={micro_step + 1}: {raw_loss}")

            loss = raw_loss_tensor / args.grad_accum
            loss.backward()
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
                    raise RuntimeError("No non-zero LoRA gradients on first microbatch.")

            should_step = (micro_step % args.grad_accum == 0) or (micro_step == len(loader) * args.epochs)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            progress.set_postfix(
                loss=f"{raw_loss:.4f}",
                updates=global_update,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    adapter_change_sq = 0.0
    for parameter, initial in zip(trainable, initial_trainable):
        delta = parameter.detach().cpu().float() - initial.float()
        adapter_change_sq += float(torch.sum(delta * delta))
    adapter_parameter_l2_change = math.sqrt(adapter_change_sq)
    if adapter_parameter_l2_change == 0.0:
        raise RuntimeError("LoRA parameters did not change.")

    metadata = {
        "model_path": str(args.model_path),
        "train_csv": str(args.train_csv),
        "train_rows": int(len(frame)),
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "choice_shuffle_prob": args.choice_shuffle_prob,
        "choice_loss": "four_choice_next_token_cross_entropy",
        "quantization": "4bit_nf4_double_quant",
        "compute_dtype": "bfloat16",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "total_updates": global_update,
        "mean_microbatch_loss": loss_sum / max(1, micro_step),
        "first_grad_norm": first_grad_norm,
        "first_grad_param_count": first_grad_param_count,
        "adapter_parameter_l2_change": adapter_parameter_l2_change,
        "permutation_counts": permutation_counts,
        "elapsed_seconds": time.time() - started,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        "seed": args.seed,
        "note": (
            "TEAM-C-inspired local 8B recipe; not an exact reproduction because TEAM-C's "
            "full adapter target/alpha details are not verified."
        ),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
