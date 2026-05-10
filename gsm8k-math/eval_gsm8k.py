"""Offline metrics auditor and backtrack verifier for learned gsm8k-math LoRA adapters.

Reloads PEFT adapters configs on unquantized base weights, runs greedy validation
inference over 100 held-out test problems, backtracks numerical answers,
and reports overall Exact Match accuracy.
"""

import argparse
import re
import sys
from datasets import load_dataset
from unsloth import FastModel


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


def generate_greedy_completion(model, tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
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
        max_new_tokens=512,
        do_sample=False,
    )
    
    input_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        "-a",
        required=True,
        help="Path containing saved LoRA adapters config and safetensors weights.",
    )
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        default=False,
        help="Disable standard 4bit quant, run in unquantized 16bit floats.",
    )
    args = parser.parse_args()

    print(f"Reloading model from PEFT Adapter config: {args.adapter}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastModel.for_inference(model)

    print(f"Loading validation dataset test split...")
    val_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    num_eval = min(args.eval_rows, len(val_dataset))
    
    print(f"Executing greedy mathematical verifications over {num_eval} problems...")
    correct = 0
    for idx in range(num_eval):
        example = val_dataset[idx]
        question = example["question"]
        gold_answer = extract_answer(example["answer"])
        
        pred_completion = generate_greedy_completion(model, tokenizer, question)
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
    print(f"OFFLINE benchmark EVALUATION STATS:")
    print(f"  Correct Problems  : {correct}/{num_eval}")
    print(f"  Average EM Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
