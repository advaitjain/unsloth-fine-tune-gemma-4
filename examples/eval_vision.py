"""Evaluate a Gemma 4 Vision checkpoint on the LaTeX OCR test split.

Loads a base model or base + adapter, and runs a deterministic (greedy)
evaluation on a representative subset of the LaTeX OCR test split.
Computes both Normalized Exact Match (EM) and Levenshtein-based Edit Distance (NED) scores.

Usage:
    uv run python examples/eval_vision.py --model lora_vision_v1
"""

import argparse
import os
import re
import torch
from PIL import Image
from unsloth import FastModel
from datasets import load_dataset


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "unsloth/LaTeX_OCR"
INSTRUCTION = "Write the LaTeX representation for this image."


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_latex(text: str) -> str:
    # 1. Extract raw LaTeX from possible wrappers
    match_code = re.search(r"```latex\s*(.*?)\s*```", text, re.DOTALL)
    if match_code:
        text = match_code.group(1)
    else:
        match_display = re.search(r"\$\$\s*(.*?)\s*\$\$", text, re.DOTALL)
        if match_display:
            text = match_display.group(1)
        else:
            match_inline = re.search(r"\$\s*(.*?)\s*\$", text, re.DOTALL)
            if match_inline:
                text = match_inline.group(1)

    # 2. Remove all spaces
    text = re.sub(r"\s+", "", text)

    # 3. Standardize common math symbols using negative lookahead to prevent corrupting \leq
    text = re.sub(r"\\le(?!q)", r"\\leq", text)
    text = re.sub(r"\\ge(?!q)", r"\\geq", text)
    text = re.sub(r"\\to", r"\\rightarrow", text)
    text = re.sub(r"\\epsilon", r"\\varepsilon", text)

    # 4. Standardize single-character or single-command subscript/superscript brace wrapping
    # e.g., x_i -> x_{i}, x_\pm -> x_{\pm}
    text = re.sub(r"_([a-zA-Z0-9]|\\[a-zA-Z]+)", r"_{\1}", text)
    text = re.sub(r"\^([a-zA-Z0-9]|\\[a-zA-Z]+)", r"^{\1}", text)

    return text.strip()


def compute_scores(pred: str, true: str) -> tuple[float, float]:
    clean_pred = normalize_latex(pred)
    clean_true = normalize_latex(true)

    # 1. Exact Match
    em = 1.0 if clean_pred == clean_true else 0.0

    # 2. Normalized Edit Distance
    max_len = max(len(clean_pred), len(clean_true))
    if max_len == 0:
        ned = 1.0
    else:
        dist = levenshtein_distance(clean_pred, clean_true)
        ned = 1.0 - (dist / max_len)

    return em, ned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id or local adapter path to evaluate (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--eval-rows",
        type=int,
        default=30,
        help="Number of test samples to evaluate (default: 30).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    parser.add_argument(
        "--vision-tokens",
        type=int,
        default=280,
        help="Visual token budget per image (e.g. 280, 560, 1120).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    # Load model and processor
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=2048,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    if args.vision_tokens != 280:
        print(f"Overriding visual token budget dynamically to {args.vision_tokens}...")
        # 1. Modify Model Config
        model.config.vision_soft_tokens_per_image = args.vision_tokens
        model.config.vision_config.default_output_length = args.vision_tokens

        # 2. Modify Processor Config
        tokenizer.image_processor.image_seq_length = args.vision_tokens
        tokenizer.image_processor.max_soft_tokens = args.vision_tokens
        if hasattr(tokenizer, "image_seq_length"):
            tokenizer.image_seq_length = args.vision_tokens

    # Put model in inference mode
    FastModel.for_inference(model)

    is_vlm = hasattr(tokenizer, "image_processor")
    if not is_vlm:
        raise ValueError(f"Model {args.model} is not a vision model!")

    print(f"Loading dataset {DATASET_NAME} split=test...")
    test_dataset = load_dataset(DATASET_NAME, split="test")

    # Select deterministic evaluation set (first N samples)
    eval_dataset = test_dataset.select(range(min(args.eval_rows, len(test_dataset))))

    total_em = 0.0
    total_ned = 0.0
    count = len(eval_dataset)

    print(f"\nStarting evaluation on {count} samples...")
    print(f"{'=' * 80}")

    for i in range(count):
        sample = eval_dataset[i]
        image = sample["image"].convert("RGB")
        ground_truth = sample["text"]

        # Generate
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": INSTRUCTION}
                ]
            }
        ]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(text=[prompt_text], images=[image], return_tensors="pt").to("cuda")

        # We use greedy decoding for deterministic evaluation
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        # Extract generated text
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        pred_text = tokenizer.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Compute scores
        em, ned = compute_scores(pred_text, ground_truth)
        total_em += em
        total_ned += ned

        print(f"\n--- Sample {i+1} ---")
        print(f"GT:   {ground_truth}")
        print(f"Pred: {pred_text}")
        print(f"Clean GT:   {normalize_latex(ground_truth)}")
        print(f"Clean Pred: {normalize_latex(pred_text)}")
        print(f"EM: {em} | NED: {ned:.4f}")

    avg_em = total_em / count
    avg_ned = total_ned / count

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Model: {args.model}")
    print(f"Precision: {'4-bit' if args.load_in_4bit else 'fp16'}")
    print(f"Total Samples: {count}")
    print(f"Average Exact Match (EM):        {avg_em * 100:.2f}%")
    print(f"Average Normalized Edit Dist (NED): {avg_ned * 100:.2f}%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
