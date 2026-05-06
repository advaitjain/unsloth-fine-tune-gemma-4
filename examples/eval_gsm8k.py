"""Evaluate a Gemma checkpoint on the fixed GSM8K eval set.

Loads either a base model alone, or a base + saved LoRA adapter (by passing
``--adapter <path>``), and runs the same three GSM8K test problems used by
``finetune_gsm8k.py`` with greedy decoding. Useful for comparing checkpoints
without re-running training.

    # Base model only (with greedy)
    uv run python examples/eval_gsm8k.py

    # Saved adapter
    uv run python examples/eval_gsm8k.py --adapter lora_gsm8k/
"""

import argparse

from unsloth import FastModel

from finetune_gsm8k import EVAL_PROMPTS, DEFAULT_MODEL, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        default=None,
        help="Path to a saved LoRA adapter dir. If set, base model is "
        "inferred from the adapter's adapter_config.json.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    target = args.adapter if args.adapter else args.model
    model, tokenizer = FastModel.from_pretrained(
        model_name=target,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    label = args.model + (f" + adapter:{args.adapter}" if args.adapter else " (base)")
    print(f"\n{'=' * 60}\nEVAL: {label}\n{'=' * 60}")
    for i, prompt in enumerate(EVAL_PROMPTS, start=1):
        print(f"\n--- Problem {i} ---\nQ: {prompt}\nA: ", end="", flush=True)
        generate(model, tokenizer, prompt, args.max_new_tokens)
        print()


if __name__ == "__main__":
    main()
