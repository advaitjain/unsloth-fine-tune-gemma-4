"""Side-by-side demonstrator checking SFT math optimizations over 5 problems.

Loads the raw base model first in 16-bit, executes Zero-Shot inferences, clears VRAM,
then reloads the fine-tuned PEFT adapter in 16-bit to execute optimized inferences.
Outputs side-by-side logical reasoning steps and answers.
"""

import argparse
import gc
import json
import re
import sys
import torch
from datasets import load_dataset
from unsloth import FastModel


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
BASE_MODEL = "unsloth/gemma-4-E2B-it"


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


def load_demonstration_problems() -> list[dict]:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    
    samples = []
    # We target specifically indices 0 to 5 which holds canonical math reasoning targets
    # (Janet's ducks, Robe bolts, Josh's flip net profit, etc.)
    for idx in range(5):
        example = dataset[idx]
        samples.append({
            "idx": idx,
            "question": example["question"],
            "answer": extract_answer(example["answer"])
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        "-a",
        required=False,
        default=None,
        help="Path to the trained PEFT adapter weights directory (optional).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        default=False,  # Force high-fidelity 16bit precision comparative checks
        help="Run demonstrator calculations in unquantized 16bit precision.",
    )
    args = parser.parse_args()

    targets = load_demonstration_problems()

    # ==========================================
    # PHASE 1: BASE ZERO-SHOT RUN
    # ==========================================
    print(f"Loading raw baseline math model: {BASE_MODEL}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastModel.for_inference(model)

    predictions_base = []
    print("Executing baseline zero-shot math predictions...")
    for idx, sample in enumerate(targets):
        completion = generate_greedy_completion(model, tokenizer, sample["question"])
        ans = extract_answer(completion)
        predictions_base.append({"text": completion, "ans": ans})
        print(f"  [{idx+1}] Compiled base baseline answer.")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Base model purged. VRAM cleared.")

    # ==========================================
    # PHASE 2: OPTIMIZED SFT RUN
    # ==========================================
    predictions_sft = []
    if args.adapter:
        print(f"\nLoading fine-tuned PEFT adapter: {args.adapter}...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        FastModel.for_inference(model)

        print("Executing SFT math predictions...")
        for idx, sample in enumerate(targets):
            completion = generate_greedy_completion(model, tokenizer, sample["question"])
            ans = extract_answer(completion)
            predictions_sft.append({"text": completion, "ans": ans})
            print(f"  [{idx+1}] Compiled SFT prediction.")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # PHASE 3: MATH REASONING COMPARISON REPORT
    # ==========================================
    print(f"\n{'=' * 80}")
    print("           GSM8K-MATH SFT DEMONSTRATOR COMPARISON REPORT")
    print(f"{'=' * 80}")

    for idx, sample in enumerate(targets):
        p_base = predictions_base[idx]
        
        print(f"\n\n### [{idx+1}] PROBLEM {idx+1}:")
        print(f"{'-' * 80}")
        print(f"Q: {sample['question'].strip()}")
        print(f"\nGOLD TARGET ANSWER: {repr(sample['answer'])}")
        
        print(f"\n--- [BEFORE SFT] Zero-Shot Base Generation (Is_Correct: {p_base['ans'] == sample['answer']}) ---")
        print(f"Extracted value: {repr(p_base['ans'])}")
        if args.adapter:
            # Print last 250 chars of base reasoning for format inspect
            print(f"Prose logic (suffix): ...{p_base['text'][-250:].replace(chr(10), ' ')}")
            
            p_sft = predictions_sft[idx]
            print(f"\n--- [AFTER SFT] Fine-Tuned LoRA Generation (Is_Correct: {p_sft['ans'] == sample['answer']}) ---")
            print(f"Extracted value: {repr(p_sft['ans'])}")
            print(f"Optimized SFT logic:\n{p_sft['text']}")
        else:
            print(f"Prose logic:\n{p_base['text']}")
            
        print(f"{'=' * 80}")

    print("\nVisual mathematical audit complete.")


if __name__ == "__main__":
    main()
