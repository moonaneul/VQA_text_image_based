from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path

import torch


def module_version(name: str):
    try:
        module = importlib.import_module(name)
        return {"installed": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"installed": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight LoRA/QLoRA support on the local VQA environment.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("downloads/models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--test-4bit-load", action="store_true")
    args = parser.parse_args()

    result = {
        "python_packages": {
            name: module_version(name)
            for name in ["transformers", "peft", "bitsandbytes", "accelerate"]
        },
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "four_bit_load": "not_tested",
    }

    required = result["python_packages"]
    if args.test_4bit_load:
        if not required["bitsandbytes"]["installed"] or not required["peft"]["installed"]:
            result["four_bit_load"] = "skipped_missing_package"
        else:
            try:
                from transformers import BitsAndBytesConfig
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
                except ImportError:
                    from transformers import AutoModelForImageTextToText as ModelClass

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = ModelClass.from_pretrained(
                    args.model_path,
                    quantization_config=quant,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                )
                result["four_bit_load"] = "success"
                result["peak_vram_gb"] = (
                    float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                    if torch.cuda.is_available()
                    else None
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                result["four_bit_load"] = "failed"
                result["four_bit_error"] = repr(exc)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
