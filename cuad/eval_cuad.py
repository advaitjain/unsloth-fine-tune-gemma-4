"""Offline benchmark tool for verifying SFT LoRA adapters.

Reloads saved PEFT weights over the baseline model, processes validation contracts,
executes greedy decoding, and reports final comparative statistics.
"""

import argparse
import json
import re
import sys
from huggingface_hub import hf_hub_download

from unsloth import FastModel


CUAD_REPO = "theatticusproject/cuad"
CUAD_FILE = "CUAD_v1/CUAD_v1.json"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.')
    return text


def calculate_metrics(prediction: str, reference: str) -> tuple[int, float]:
    p_norm = normalize_text(prediction)
    r_norm = normalize_text(reference)
    
    em = 1 if p_norm == r_norm else 0
    
    pred_tokens = p_norm.split()
    ref_tokens = r_norm.split()
    
    if not pred_tokens or not ref_tokens:
        f1 = 1.0 if pred_tokens == ref_tokens else 0.0
        return em, f1
        
    common = set(pred_tokens) & set(ref_tokens)
    num_same = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)
    
    if num_same == 0:
        return em, 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return em, f1


def load_cuad_val_split(eval_rows: int) -> list[dict]:
    print("Loading validation subset from raw JSON annotations...")
    local_file = hf_hub_download(
        repo_id=CUAD_REPO,
        filename=CUAD_FILE,
        repo_type="dataset"
    )
    with open(local_file, "r") as f:
        data = json.load(f)
        
    contracts = data.get("data", [])
    
    samples = []
    for contract in contracts:
        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue
        context = paragraphs[0].get("context", "")
        qas = paragraphs[0].get("qas", [])
        
        gov_law_qa = next((qa for qa in qas if qa['id'].endswith('__Governing Law')), None)
        if gov_law_qa and len(gov_law_qa.get('answers', [])) > 0:
            answer = gov_law_qa['answers'][0]
            ans_text = answer['text']
            ans_start = answer['answer_start']
            
            # Crop exact context
            start_idx = max(0, ans_start - 256)
            end_idx = min(len(context), ans_start + len(ans_text) + 256)
            snippet = context[start_idx:end_idx]
            
            prompt = (
                "Extract the precise sentence or section of the contract text below that specifies "
                "the Governing Law of this agreement. If the Governing Law is not specified, respond only with 'None'.\n\n"
                f"Snippet:\n\"\"\"\n{snippet.strip()}\n\"\"\""
            )
            
            samples.append({
                "question": prompt,
                "answer": ans_text
            })
            
    train_count = 300
    val_count = min(eval_rows, len(samples) - train_count)
    val_split = samples[train_count:train_count + val_count]
    
    print(f"Extracted {len(val_split)} evaluation records from validation dataset split.")
    return val_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        "-a",
        required=True,
        help="Path containing local saved adapters and configs.",
    )
    parser.add_argument("--eval-rows", type=int, default=50)
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Run operations using unquantized 16-bit floats.",
    )
    args = parser.parse_args()

    print(f"Loading base model with Peft adapter: {args.adapter}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastModel.for_inference(model)  # optimized routines

    val_split = load_cuad_val_split(args.eval_rows)

    total_em = 0.0
    total_f1 = 0.0
    
    print(f"\nEvaluating offline checkpoints on validation split (greedy)...")
    for i, sample in enumerate(val_split):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": sample["question"]}],
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
            max_new_tokens=256,
            do_sample=False,
        )
        
        input_len = inputs.input_ids.shape[1]
        pred_ids = outputs[0][input_len:]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        
        em, f1 = calculate_metrics(pred_text, sample["answer"])
        total_em += em
        total_f1 += f1
        
        if i < 5:
            print(f" [{i+1:2d}] Pred: {repr(pred_text)}")
            print(f"      Gold: {repr(sample['answer'])}")
            print(f"      Scores: EM={em}, F1={f1:.3f}")
            
    avg_em = (total_em / len(val_split)) * 100
    avg_f1 = (total_f1 / len(val_split)) * 100
    print(f"\nFINAL STATS REPORT (Greedy over {len(val_split)} validation items):")
    print(f"  Exact Match (EM)  : {avg_em:.2f}%")
    print(f"  Token-level F1    : {avg_f1:.2f}%")


if __name__ == "__main__":
    main()
