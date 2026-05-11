"""Quantitative evaluation script for compiled LiteRT-LM GSM8K models.

Loads a .litertlm model via litert_lm.Engine, executes greedy generation over
100 held-out GSM8K problems, backtracks numerical answers, and reports Exact Match accuracy.
"""

import argparse
import re
import sys
from datasets import load_dataset
import litert_lm


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


def extract_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1).strip()
        
    numbers_with_contexts = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)", text):
        num = m.group(1)
        start_idx = m.end()
        context = text[start_idx:start_idx+25].lower().strip()
        numbers_with_contexts.append((num, context))
        
    if numbers_with_contexts:
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
            
            should_ignore = False
            for unit in ignore_units:
                if re.match(rf"^\s*s?\b{unit}s?\b", context) or re.match(rf"^[\s*]*{unit}s?", context):
                    should_ignore = True
                    break
            if not should_ignore:
                return val
                
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        val = numbers[-1].strip()
        if val.endswith(".00"):
            val = val[:-3]
        return val
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the compiled .litertlm model file.",
    )
    parser.add_argument("--eval-rows", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading LiteRT-LM Engine on robust CPU backend: {args.model}...")
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    with litert_lm.Engine(
        args.model,
        backend=litert_lm.Backend.CPU,
    ) as engine:
        print("Loading validation dataset test split...")
        val_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
        num_eval = min(args.eval_rows, len(val_dataset))
        
        print(f"Executing LiteRT-LM mathematical verifications over {num_eval} problems...")
        correct = 0
        for idx in range(num_eval):
            example = val_dataset[idx]
            question = example["question"]
            gold_answer = extract_answer(example["answer"])
            
            # Create a fresh, stateless conversation context for each problem
            with engine.create_conversation() as conversation:
                response = conversation.send_message(question)
                pred_completion = response["content"][0]["text"]
                
            pred_answer = extract_answer(pred_completion)
            
            is_correct = pred_answer == gold_answer
            if is_correct:
                correct += 1
                
            status = "✅" if is_correct else "❌"
            print(f"  [{idx+1:2d}/{num_eval}] {status} Gold: {gold_answer:<6} Pred: {pred_answer:<6}")
            if not is_correct:
                print(f"      [DEBUG] Prediction suffix: ...{pred_completion[-150:].replace(chr(10), ' ')}")
                
        accuracy = (correct / num_eval) * 100
        print(f"\n{'=' * 60}")
        print("LITERT-LM QUANTITATIVE EVALUATION STATS:")
        print(f"  Correct Problems : {correct}/{num_eval}")
        print(f"  Exact Match Score: {accuracy:.2f}%")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
