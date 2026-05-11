"""Merge the best performing GSM8K LoRA adapter into full 16-bit safetensors.

Loads the optimal adapter (EXP_17), merges it with base weights in fp16 precision,
and exports standard safetensors for LiteRT-LM compilation.
"""

import argparse
import torch
from unsloth import FastModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        "-a",
        default="gsm8k-math/lora_exp17",
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="litert-lm/merged_model",
        help="Directory to save the merged fp16 safetensors.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading adapter from {args.adapter} in fp16 precision...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    print(f"Merging weights and exporting to {args.output_dir}...")
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print("Adapter successfully merged and saved.")


if __name__ == "__main__":
    main()
