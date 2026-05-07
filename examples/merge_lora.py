"""Merge LoRA adapters into the base model to produce a 16-bit merged model.

Loads a saved LoRA adapter, merges it with the base model in fp16 precision
(without 4-bit quantization), and saves the merged model as a single set of
safetensors.

Usage:
    uv run python examples/merge_lora.py --adapter lora_vision_full --output-dir merged_vision
"""

import argparse
import torch
from unsloth import FastModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        "-a",
        required=True,
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to save the merged fp16 model.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading adapter from {args.adapter} in fp16...")
    # Load model and tokenizer/processor.
    # We use FastModel as it dynamically handles both text and vision models.
    # We set load_in_4bit=False and dtype=torch.float16 for fp16 merge.
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    print(f"Merging weights and saving to {args.output_dir} in 16-bit...")
    # save_pretrained_merged will merge the LoRA weights into the base model
    # and save it. We use tokenizer here (which might be a processor for VLM).
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Successfully merged and saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
