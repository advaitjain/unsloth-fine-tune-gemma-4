"""Minimal Unsloth + Gemma fine-tuning example on GSM8K.

Demonstrates the effect of supervised fine-tuning by running a fixed set of
held-out math problems through the model BEFORE training, fine-tuning on a
GSM8K subset, then running the same problems AFTER. The format imprint is
the headline: base outputs are prose paragraphs; post-fine-tune outputs end
with the GSM8K-style ``#### <answer>`` marker.

The default model is Gemma 3 1B (4-bit) which peaks around ~3 GB of VRAM
during QLoRA training at the default settings — fits comfortably on a 6 GB
GPU. Override with --model to fine-tune a different Gemma checkpoint:

    uv run python examples/finetune_gsm8k.py --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit
"""

import argparse

from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


DEFAULT_MODEL = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"

# Three GSM8K *test* problems — held out from the train split we fine-tune on.
EVAL_PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
    "many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and "
    "then puts in $50,000 in repairs. This increased the value of the house "
    "by 150%. How much profit did he make?",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> None:
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
    # Greedy decoding for math: removes sampling noise so each computation
    # is deterministic. Sampling (temperature=1.0) tends to drift mid-chain
    # and produce inconsistent arithmetic.
    model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def format_example(example: dict, tokenizer) -> str:
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


def run_eval(model, tokenizer, banner: str) -> None:
    print(f"\n{'=' * 60}\n{banner}\n{'=' * 60}")
    for i, prompt in enumerate(EVAL_PROMPTS, start=1):
        print(f"\n--- Problem {i} ---\nQ: {prompt}\nA: ", end="", flush=True)
        generate(model, tokenizer, prompt)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id to fine-tune (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--train-rows",
        type=int,
        default=1000,
        help="Subset of GSM8K train to use; 0 means full split (~7.5k).",
    )
    parser.add_argument("--output-dir", default="lora_gsm8k")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
        help="LoRA alpha. Common choices: equal to rank, or 2x rank.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="HF scheduler name: linear, cosine, constant, etc.",
    )
    args = parser.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

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

    run_eval(model, tokenizer, "BEFORE FINE-TUNING")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    if args.train_rows > 0:
        dataset = dataset.select(range(min(args.train_rows, len(dataset))))
    dataset = dataset.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
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

    # Mask question tokens so loss is computed only on the model's answer.
    # Gemma 4 uses different turn markers than Gemma 3.
    if "gemma-4" in args.model.lower():
        instruction_part = "<|turn>user\n"
        response_part = "<|turn>model\n"
    else:
        instruction_part = "<start_of_turn>user\n"
        response_part = "<start_of_turn>model\n"

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    trainer.train()

    run_eval(model, tokenizer, "AFTER FINE-TUNING")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
