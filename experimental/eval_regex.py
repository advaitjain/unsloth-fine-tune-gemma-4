"""Automated evaluation of Gemma checkpoints on the NL-to-Regex task.

Loads a base model or base + saved LoRA adapter, runs greedy decoding on the first
N examples of the nl-to-regex dataset, and calculates Exact Match (EM) accuracy.

Example:
    # Evaluate base 4-bit model
    uv run python experimental/eval_regex.py --num-examples 50

    # Evaluate a saved adapter
    uv run python experimental/eval_regex.py --adapter experimental/lora_regex/ --num-examples 50
"""

import argparse
import re
from datasets import load_dataset
from unsloth import FastModel
import torch

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "inclinedadarsh/nl-to-regex"


def extract_regex_from_completion(text: str) -> str:
    # 1. Look for fenced code blocks: ```regex ... ``` or ``` ... ```
    match = re.search(r"```(?:regex)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # 2. Look for inline backtick wraps: `...`
    match = re.search(r"`(.*?)`", text)
    if match:
        return match.group(1).strip()
        
    # 3. Look for bolded pattern declarations: **Regex:** pattern
    match = re.search(r"\*\*Regex:\*\*\s*([^\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    # 4. Default: Fallback to raw cleaned text
    return text.strip()


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Generate a regular expression for the following pattern:\n{prompt}",
                }
            ],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


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
        help="Path to a saved LoRA adapter dir. If set, base model is inferred from adapter_config.json.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of test examples to evaluate (default: 50).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    target = args.adapter if args.adapter else args.model
    print(f"Loading model: {target} (load_in_4bit={args.load_in_4bit})")

    model, tokenizer = FastModel.from_pretrained(
        model_name=target,
        dtype=None,  # Unsloth auto-detects
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # Set evaluation mode
    FastModel.for_inference(model)

    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    num_eval = min(args.num_examples, len(dataset))
    print(f"Evaluating first {num_eval} examples...")

    correct = 0
    for idx in range(num_eval):
        example = dataset[idx]
        nl_prompt = example["user"]
        gold_regex = example["assistant"].strip()

        pred_completion = generate_completion(model, tokenizer, nl_prompt, args.max_new_tokens)
        
        # Exact match after extracting patterns from markdown wraps
        pred_extracted = extract_regex_from_completion(pred_completion)
        gold_extracted = extract_regex_from_completion(gold_regex)

        pred_cleaned = pred_extracted.strip().strip("'").strip('"')
        gold_cleaned = gold_extracted.strip().strip("'").strip('"')

        is_correct = pred_cleaned == gold_cleaned
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"[{idx + 1}/{num_eval}] {status}")
        print(f"  Prompt: {nl_prompt}")
        print(f"  Gold  : {gold_cleaned}")
        print(f"  Pred  : {pred_cleaned}")

    accuracy = (correct / num_eval) * 100
    print(f"\n{'=' * 40}")
    print(f"Exact Match Accuracy: {correct}/{num_eval} ({accuracy:.2f}%)")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
