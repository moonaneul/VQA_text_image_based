from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect how the local Qwen tokenizer represents multiple-choice labels."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"),
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    tokenizer = processor.tokenizer

    variants = [
        "a", "b", "c", "d",
        " a", " b", " c", " d",
        "\na", "\nb", "\nc", "\nd",
    ]

    print("=== Choice tokenization ===")
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"{text!r:>6} -> ids={ids}, tokens={tokens}, n_tokens={len(ids)}")

    print("\n=== Minimal assistant-context suffix check ===")
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": "정답은 a, b, c, d 중 하나만 출력하세요."}],
    }]
    prefix = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    for choice in "abcd":
        full_ids = tokenizer.encode(prefix + choice, add_special_tokens=False)
        suffix = full_ids[len(prefix_ids):]
        print(
            f"choice={choice!r} -> suffix_ids={suffix}, "
            f"suffix_tokens={tokenizer.convert_ids_to_tokens(suffix)}, "
            f"n_suffix={len(suffix)}"
        )


if __name__ == "__main__":
    main()
