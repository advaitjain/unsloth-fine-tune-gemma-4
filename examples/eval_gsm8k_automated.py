"""Automated evaluation of Gemma checkpoints on the GSM8K test set.

Loads a base model or base + saved LoRA adapter, runs greedy decoding on a subset
of the GSM8K test split, extracts final numerical answers, and calculates overall accuracy.

Example:
    # Evaluate base 4-bit model
    uv run python examples/eval_gsm8k_automated.py --num-examples 50

    # Evaluate a saved adapter
    uv run python examples/eval_gsm8k_automated.py --adapter lora_gsm8k/ --num-examples 50

    # Evaluate base fp16 model
    uv run python examples/eval_gsm8k_automated.py --model unsloth/gemma-4-E2B-it --no-4bit --num-examples 50
"""

import argparse
import re
from datasets import load_dataset
from unsloth import FastModel
import torch

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


def extract_answer(text: str) -> str:
    # Remove commas from numbers in the text (e.g., "70,000" -> "70000")
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    
    # 1. Standard GSM8K format: #### <answer>
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1).strip()
    
    # 2. Context-aware number extraction: find numbers and check their subsequent context
    numbers_with_contexts = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)", text):
        num = m.group(1)
        start_idx = m.end()
        context = text[start_idx:start_idx+25].lower().strip()
        numbers_with_contexts.append((num, context))
        
    if numbers_with_contexts:
        # Ignore units list
        ignore_units = ["week", "day", "hour", "minute", "second", "liter", "carton", "glass", "item", "ounce", "cleaner"]
        for num, context in reversed(numbers_with_contexts):
            val = num
            if val.endswith(".00"):
                val = val[:-3]
            elif "." in val:
                try:
                    f_val = float(val)
                    if f_val.is_integer():
                        val = str(int(f_val))
                except ValueError:
                    pass
            
            # Check if the context indicates this is a unit count, not the actual answer
            should_ignore = False
            for unit in ignore_units:
                if re.match(rf"^\s*s?\b{unit}s?\b", context) or re.match(rf"^[\s*]*{unit}s?", context):
                    should_ignore = True
                    break
            if not should_ignore:
                return val

    # 3. Fallback: find the last number (integer or decimal) in the generated response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        val = numbers[-1].strip()
        if val.endswith(".00"):
            val = val[:-3]
        return val
    return ""


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
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

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


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
    parser.add_argument("--max-new-tokens", type=int, default=512)
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

    print(f"Loading GSM8K test split...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    num_eval = min(args.num_examples, len(dataset))
    print(f"Evaluating first {num_eval} examples...")

    correct = 0
    for idx in range(num_eval):
        example = dataset[idx]
        question = example["question"]
        gold_answer = extract_answer(example["answer"])

        pred_completion = generate_completion(model, tokenizer, question, args.max_new_tokens)
        pred_answer = extract_answer(pred_completion)

        is_correct = pred_answer == gold_answer
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"[{idx + 1}/{num_eval}] {status} Gold: {gold_answer:<6} Pred: {pred_answer:<6}")
        
        # If wrong, print a small snippet of prediction for debugging
        if not is_correct:
            # print last 150 chars of prediction
            print(f"      [DEBUG] Pred (suffix): ...{pred_completion[-150:].replace(chr(10), ' ')}")

    accuracy = (correct / num_eval) * 100
    print(f"\n{'=' * 40}")
    print(f"Accuracy: {correct}/{num_eval} ({accuracy:.2f}%)")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
