"""Fine-tuning pipeline for contract clause extraction on CUAD using Gemma 4 E2B.

Downloads the official SQuAD formatted CUAD JSON data, extracts context slices,
runs zero-shot benchmarks, fine-tunes LoRA parameters, logs post-SFT metrics,
and saves the learned Peft adapter.
"""

import argparse
import json
import os
import re
import sys
from huggingface_hub import hf_hub_download

from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it"
CUAD_REPO = "theatticusproject/cuad"
CUAD_FILE = "CUAD_v1/CUAD_v1.json"


def normalize_text(text: str) -> str:
    """Standard SQuAD parsing standardization to prevent structural mismatch."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # collapse white space
    text = text.rstrip('.')
    return text


def calculate_metrics(prediction: str, reference: str) -> tuple[int, float]:
    """Compute SQuAD bag-of-words overlap F1 and Normalized Exact Match."""
    p_norm = normalize_text(prediction)
    r_norm = normalize_text(reference)
    
    # Exact Match metric
    em = 1 if p_norm == r_norm else 0
    
    # Overlap F1 metric
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


def load_cuad_samples(eval_rows: int) -> tuple[list[dict], list[dict]]:
    """Download and parse contract segments surrounding exact targeted bounds."""
    print("Downloading and loading CUAD_v1.json database...")
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
            
            # Crop offset paragraph containing exact target
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
            
    print(f"Extracted {len(samples)} contract snippet extraction datasets.")
    
    # deterministic train/validation split allocations
    train_count = 300
    val_count = min(eval_rows, len(samples) - train_count)
    
    train_split = samples[:train_count]
    val_split = samples[train_count:train_count + val_count]
    
    print(f"Allocated dataset split: {len(train_split)} train samples, {len(val_split)} validation benchmarks.")
    return train_split, val_split


def format_example(example: dict, tokenizer) -> str:
    """Wrap example inside standard gemma 4 conversation tokens."""
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": example["question"]}],
        },
        {
            "role": "model",
            "content": [{"type": "text", "text": example["answer"]}],
        },
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def run_eval(model, tokenizer, val_split: list[dict], banner: str) -> tuple[float, float]:
    """Greedy validation execution and computing scores averages."""
    print(f"\n{'=' * 60}\n{banner}\n{'=' * 60}")
    
    total_em = 0.0
    total_f1 = 0.0
    
    # Show preview generations for visual audit
    preview_samples = val_split[:5]
    
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
        
        # Greedy extraction pipeline
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        
        # Strip prompt to parse generated response only
        input_len = inputs.input_ids.shape[1]
        pred_ids = outputs[0][input_len:]
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        
        em, f1 = calculate_metrics(pred_text, sample["answer"])
        total_em += em
        total_f1 += f1
        
        if i < 5:
            print(f"\n--- Audit sample {i+1} ---")
            print(f"Context question preview: ...{sample['question'][-200:]}")
            print(f"Gold reference answer: {repr(sample['answer'])}")
            print(f"Model prediction output: {repr(pred_text)}")
            print(f"Sample Scores: EM={em}, F1={f1:.3f}")
            
    avg_em = (total_em / len(val_split)) * 100
    avg_f1 = (total_f1 / len(val_split)) * 100
    print(f"\nSummary Results for {banner}: EM={avg_em:.2f}%, Overlap F1={avg_f1:.2f}%")
    return avg_em, avg_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Unsloth compatible base model (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=50)
    parser.add_argument("--output-dir", default="cuad/lora_cuad")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="HF decay curves: linear, cosine, etc.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Load raw 16-bit precision weights directly.",
    )
    args = parser.parse_args()

    # 1. Model loading config
    print(f"Initializing base: {args.model} (load_in_4bit={args.load_in_4bit})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # 2. Inject LoRA target parameters
    print(f"Injecting Peft weights config (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 3. Load datasets splits
    train_split, val_split = load_cuad_samples(args.eval_rows)

    # 4. Run benchmark before SFT starts
    pre_em, pre_f1 = run_eval(model, tokenizer, val_split, "BEFORE FINE-TUNING (Zero-Shot baseline)")

    # 5. Process train token markers
    formatted_train = [format_example(ex, tokenizer) for ex in train_split]
    train_dataset = Dataset.from_dict({"text": formatted_train})

    # 6. Configure SFT Config loops
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    # 7. SFT Turn masking targets response parameters
    # Instruction part is standard user, Response marker triggers standard model turns for gemma-4
    instruction_part = "<|turn>user\n"
    response_part = "<|turn>model\n"
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    # 8. Start fine-tuning
    print(f"\nStarting fine-tuning iterations for {args.max_steps} steps...")
    trainer.train()

    # 9. Run benchmark after SFT finishes
    post_em, post_f1 = run_eval(model, tokenizer, val_split, "AFTER FINE-TUNING (SFT adapters benchmark)")

    # 10. Save adapter weights
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter weights exported directly to {args.output_dir}/")
    
    print(f"\nSUMMARY STATS OVERVIEW:")
    print(f"  PRE-SFT  : Exact Match = {pre_em:.2f}%, Token Overlap F1 = {pre_f1:.2f}%")
    print(f"  POST-SFT : Exact Match = {post_em:.2f}%, Token Overlap F1 = {post_f1:.2f}%")


if __name__ == "__main__":
    main()
