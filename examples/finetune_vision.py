"""Unsloth + Gemma 4 Vision fine-tuning example on LaTeX OCR.

Demonstrates fine-tuning a Gemma 4 Vision model to convert images of math formulas
into LaTeX representations. It runs a few evaluation examples BEFORE and AFTER
training to demonstrate the learning.

Gemma 4 E2B QLoRA fine-tuning requires ~8–10 GB peak VRAM in 4-bit.
The script defaults to memory-efficient settings (4-bit loading).
"""

import argparse
import torch
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "unsloth/LaTeX_OCR"
INSTRUCTION = "Write the LaTeX representation for this image."


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate(model, processor, image, instruction: str, max_new_tokens: int = 512) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    
    model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding for OCR accuracy
        streamer=TextStreamer(processor.tokenizer, skip_prompt=True),
    )


def run_eval(model, processor, eval_dataset, banner: str) -> None:
    print(f"\n{'=' * 60}\n{banner}\n{'=' * 60}")
    for i in range(min(3, len(eval_dataset))):
        sample = eval_dataset[i]
        image = sample["image"]
        ground_truth = sample["text"]
        print(f"\n--- Sample {i+1} ---")
        print(f"Expected LaTeX: {ground_truth}")
        print("Model Output:   ", end="", flush=True)
        generate(model, processor, image, INSTRUCTION)
        print()


def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": INSTRUCTION}
                ]
            },
            {
                "role": "model",
                "content": [
                    {"type": "text", "text": sample["text"]}
                ]
            }
        ]
    }


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
        help="Subset of train dataset to use; 0 means full split (~68k).",
    )
    parser.add_argument("--output-dir", default="lora_vision")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha. Defaults to 2x rank if not specified.",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="",
        help="Optional comma-separated target modules to apply LoRA to explicitly.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="HF scheduler name: linear, cosine, constant, etc.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    parser.add_argument(
        "--finetune-vision-layers",
        type=str2bool,
        default=True,
        help="Whether to fine-tune vision layers (True/False).",
    )
    parser.add_argument(
        "--finetune-language-layers",
        type=str2bool,
        default=True,
        help="Whether to fine-tune language layers (True/False).",
    )
    parser.add_argument(
        "--finetune-attention-modules",
        type=str2bool,
        default=True,
        help="Whether to fine-tune attention modules (True/False).",
    )
    parser.add_argument(
        "--finetune-mlp-modules",
        type=str2bool,
        default=True,
        help="Whether to fine-tune MLP modules (True/False).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (active when max-steps=-1).",
    )
    parser.add_argument(
        "--vision-tokens",
        type=int,
        default=280,
        help="Visual token budget per image (e.g. 280, 560, 1120).",
    )
    args = parser.parse_args()

    # Natively enforce the lora_alpha = 2 * lora_rank rule if omitted
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
        print(f"Enforcing default 2x Alpha rule: setting lora_alpha = {args.lora_alpha}")

    # Resolve custom target modules
    target_modules_list = None
    if args.target_modules.strip():
        target_modules_list = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        print(f"Explicitly targeting modules: {target_modules_list}")

    # Load model and processor
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    if args.vision_tokens != 280:
        print(f"Overriding visual token budget dynamically to {args.vision_tokens}...")
        # 1. Modify Model Config
        model.config.vision_soft_tokens_per_image = args.vision_tokens
        model.config.vision_config.default_output_length = args.vision_tokens

        # 2. Modify Processor Config
        processor.image_processor.image_seq_length = args.vision_tokens
        processor.image_processor.max_soft_tokens = args.vision_tokens
        if hasattr(processor, "image_seq_length"):
            processor.image_seq_length = args.vision_tokens

    # Configure LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        target_modules=target_modules_list,
    )

    # Load datasets
    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    test_dataset = load_dataset(DATASET_NAME, split="test")

    # Select eval samples from raw test dataset
    eval_samples = test_dataset.select(range(3))

    # Prepare training dataset
    if args.train_rows > 0:
        train_dataset = dataset.select(range(min(args.train_rows, len(dataset))))
    else:
        train_dataset = dataset

    print(f"Converting {len(train_dataset)} training samples to conversation format...")
    converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]

    # Apply chat template to processor
    processor = get_chat_template(processor, "gemma-4")

    # Run BEFORE evaluation
    run_eval(model, processor, eval_samples, "BEFORE FINE-TUNING")

    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        processor=processor,
        data_collator=UnslothVisionDataCollator(model=model, processor=processor),
        train_dataset=converted_train_dataset,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        args=SFTConfig(
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.epochs if args.max_steps == -1 else 1.0,
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

    # Train
    print("Starting training...")
    trainer.train()

    # Run AFTER evaluation
    run_eval(model, processor, eval_samples, "AFTER FINE-TUNING")

    # Save
    print(f"Saving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter and processor saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
