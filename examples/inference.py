"""Minimal Unsloth + Gemma inference smoke test.

Loads a 4-bit instruction-tuned Gemma model and streams a single response,
to confirm the environment is wired up correctly before moving on to
fine-tuning examples.

The default model is Gemma 4 E2B (recommended target for this repo) and
needs ~8 GB of VRAM. On smaller cards, override with --model, for example:

    uv run python examples/inference.py --model unsloth/gemma-3-1b-it-unsloth-bnb-4bit
"""

import argparse

from unsloth import FastModel
from transformers import TextStreamer


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DEFAULT_PROMPT = "What is the capital of France?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help=f"User prompt to send to the model (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id to load (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    args = parser.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
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
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


if __name__ == "__main__":
    main()
