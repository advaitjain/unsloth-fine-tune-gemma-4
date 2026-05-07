"""SFT QLoRA Fine-Tuning of Gemma 4 E2B on the Emotion Classification task.

Aligns boundary weights on the dair-ai/emotion dataset using low-learning-rate
SFT sweeps, allowing conversational explanations while maximizing parsed accuracy.
"""

import argparse
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "dair-ai/emotion"
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def format_example(example: dict, tokenizer) -> str:
    gold_emotion = EMOTIONS[example["label"]]
    prompt_instruction = (
        "Classify the following diary entry into one of these 6 emotions: "
        "sadness, joy, love, anger, fear, surprise.\n\n"
        f"Diary Entry: \"{example['text']}\"\n\n"
        "Provide a brief conversational explanation of your reasoning, "
        "then output your final category strictly inside the tag: **Emotion:** <category>"
    )
    
    # We SFT the model to output a concise logical reasoning followed by the tag
    target_response = (
        f"The diary entry expresses elements associated with {gold_emotion}. "
        f"**Emotion:** {gold_emotion}"
    )

    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_instruction}],
        },
        {
            "role": "model",
            "content": [{"type": "text", "text": target_response}],
        },
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Base model id to fine-tune (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output-dir", default="experimental/lora_emotion")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--lr-scheduler-type",
        default="cosine",
        help="HF scheduler name: linear, cosine, constant, etc.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=2000,
        help="Number of training rows (default: 2000).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model} (load_in_4bit={args.load_in_4bit})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
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

    print(f"Loading and preparing dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    train_size = min(args.train_rows, len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    print(f"Selected training rows: {len(train_dataset)}")

    train_dataset = train_dataset.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        remove_columns=train_dataset.column_names,
    )

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

    # Apply token masking to calculate loss strictly on model responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print("Starting training...")
    trainer.train()

    # Guarantee weights are cleanly saved
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved successfully to {args.output_dir}/")


if __name__ == "__main__":
    main()
