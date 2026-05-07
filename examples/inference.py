"""Minimal Unsloth + Gemma inference smoke test.

Loads a 4-bit instruction-tuned Gemma model and streams a single response,
to confirm the environment is wired up correctly before moving on to
fine-tuning examples.

The default model is Gemma 4 E2B (recommended target for this repo) and
needs ~8 GB of VRAM. On smaller cards, override with --model, for example:

    uv run python examples/inference.py --model unsloth/gemma-3-1b-it-unsloth-bnb-4bit
"""

import argparse
import os
from PIL import Image

from unsloth import FastModel
from transformers import TextStreamer

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id to load (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="User prompt. Defaults to VLM prompt if image is used, else general Q&A.",
    )
    parser.add_argument(
        "--image",
        "-i",
        default="examples/sample_latex.png",
        help="Path to image file for vision inference (default: examples/sample_latex.png).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    args = parser.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # Try to load image if VLM is supported and image exists
    image = None
    is_vlm = hasattr(tokenizer, "image_processor")

    if is_vlm and args.image and os.path.exists(args.image):
        try:
            image = Image.open(args.image).convert("RGB")
            print(f"Loaded image from {args.image}")
        except Exception as e:
            print(f"Warning: Failed to load image from {args.image}: {e}. Falling back to text-only.")
    elif args.image and not os.path.exists(args.image) and args.image != "examples/sample_latex.png":
        print(f"Warning: Image file {args.image} not found. Falling back to text-only.")

    # Resolve prompt
    if args.prompt is None:
        if image is not None:
            prompt = "Write the LaTeX representation for this image."
        else:
            prompt = "What is the capital of France?"
    else:
        prompt = args.prompt

    print(f"Prompt: {prompt}")

    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # tokenizer is a processor here
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(text=[prompt_text], images=[image], return_tensors="pt").to("cuda")
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to("cuda")

    # Gemma team's recommended sampling settings.
    model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer.tokenizer if is_vlm else tokenizer, skip_prompt=True),
    )


if __name__ == "__main__":
    main()
