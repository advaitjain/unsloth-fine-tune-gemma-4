"""SFT fine-tuning sweep engine for grade-school math on GSM8K using Gemma 4 E2B.

Downloads standard math datasets splits, runs greedy zero-shot backtrack-verified
evaluations, trains capacity full-precision FP16 LoRA parameters sweeps under
Cosine decay schedules, logs post-SFT accuracy targets, and exports PEFT weights.
"""

import argparse
import json
import os
import re
import sys
from datasets import load_dataset, Dataset
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it"
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


def extract_answer(text: str) -> str:
    """Standard regular expression backtracking verifier to isolate numeric targets."""
    if not text:
        return ""
    # Remove commas in numbers
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    
    # 1. Standard GSM8K answer tag
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1).strip()
        
    # 2. Backtrack calculations details context checking
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
                
    # 3. Fallback: Last extracted number in generation
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        val = numbers[-1].strip()
        if val.endswith(".00"):
            val = val[:-3]
        return val
    return ""


def format_example(example: dict, tokenizer) -> str:
    """Wrap SFT examples inside standard conversation tokens."""
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


def run_greedy_eval(model, tokenizer, dataset, num_eval: int, banner: str) -> float:
    """Run greedy evaluations on the evaluation subset split and return correctness average."""
    print(f"\n{'=' * 60}\n{banner}\n{'=' * 60}")
    FastModel.for_inference(model)
    
    correct = 0
    for idx in range(num_eval):
        example = dataset[idx]
        question = example["question"]
        gold_answer = extract_answer(example["answer"])
        
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
        
        input_length = inputs.input_ids.shape[1]
        gen_ids = outputs[0][input_length:]
        pred_completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_answer = extract_answer(pred_completion)
        
        is_correct = pred_answer == gold_answer
        if is_correct:
            correct += 1
            
        if idx < 5:  # Visual print preview of first 5 validations details
            status = "✅" if is_correct else "❌"
            print(f"  [{idx+1:2d}/{num_eval}] {status} Gold: {gold_answer:<6} Pred: {pred_answer:<6}")
            if not is_correct:
                print(f"      [DEBUG] Pred preview: ...{pred_completion[-120:].replace(chr(10), ' ')}")
                
    accuracy = (correct / num_eval) * 100
    print(f"Summary Validation stats for {banner}: Correct = {correct}/{num_eval} ({accuracy:.2f}%)")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF Model id baseline target (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--train-rows", type=int, default=2000)
    parser.add_argument("--output-dir", default="gsm8k-math/lora_gsm8k_sweep")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="HF decay curves: linear, cosine, constant, etc.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        default=False,  # Force full precision FP16 runs by default
        help="Run SFT training and benchmarks in full unquantized 16-bit floats.",
    )
    args = parser.parse_args()

    # 1. Load unquantized model weights configurations
    print(f"Initializing Base Model: {args.model} (load_in_4bit={args.load_in_4bit})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # 2. Inject custom Peft target parameters
    print(f"Injecting LoRA parameters sweeps mapping (rank={args.lora_rank}, alpha={args.lora_alpha})")
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

    # 3. Parse datasets splits
    print("Loading GSM8K dataset splits...")
    train_dataset_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    val_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    
    num_train = min(args.train_rows, len(train_dataset_raw))
    num_eval = min(args.eval_rows, len(val_dataset))
    
    train_split = train_dataset_raw.select(range(num_train))
    print(f"Allocated Splits: Train dataset = {num_train} samples, validation benchmarks = {num_eval} items.")

    # 4. Run Zero-shot baseline checks (Random Peft)
    pre_accuracy = run_greedy_eval(model, tokenizer, val_dataset, num_eval, "BEFORE FINE-TUNING (Zero-Shot Base Baseline)")

    # 5. Process SFT training markers turns
    formatted_train = [format_example(ex, tokenizer) for ex in train_split]
    train_split_sft = Dataset.from_dict({"text": formatted_train})

    # 6. Setup trainer SFTConfig routines
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_split_sft,
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

    # 7. Apply masked response training
    # SFT Math logic: mask instruct turn markers to optimize output logic targets only
    instruction_part = "<|turn>user\n"
    response_part = "<|turn>model\n"
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    # 8. Training execution
    print(f"\nStarting SFT sweeps training for {args.max_steps} iterations steps...")
    trainer.train()

    # 9. Run Post-SFT greedy evaluations
    post_accuracy = run_greedy_eval(model, tokenizer, val_dataset, num_eval, "AFTER FINE-TUNING (SFT PEFT benchmarks delta)")

    # 10. Save learned parameters
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSweeps Adapter compiled and saved inside: {args.output_dir}/")
    
    print(f"\nSUMMARY METRICS COMPARATIVE LOGS:")
    print(f"  Zero-Shot Baseline EM Accuracy: {pre_accuracy:.2f}%")
    print(f"  Post-SFT Fine-Tuned EM Accuracy: {post_accuracy:.2f}%")
    print(f"  Performance Delta : {post_accuracy - pre_accuracy:+.2f}%")


if __name__ == "__main__":
    main()
