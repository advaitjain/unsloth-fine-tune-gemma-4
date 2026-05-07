"""Automated evaluation of Gemma checkpoints on the Emotion Classification task.

Loads a base model or base + saved LoRA adapter, runs greedy decoding on the first
N examples of the dair-ai/emotion test split, and calculates classification accuracy.

Example:
    # Evaluate base 4-bit model
    uv run python experimental/eval_emotion.py --num-examples 50

    # Evaluate a saved adapter
    uv run python experimental/eval_emotion.py --adapter experimental/lora_emotion/ --num-examples 50
"""

import argparse
import re
from datasets import load_dataset
from unsloth import FastModel
import torch

DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATASET_NAME = "dair-ai/emotion"

EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def extract_emotion(text: str) -> str:
    text_lower = text.lower()
    # 1. Check for standard **Emotion:** <category>
    match = re.search(r"\*\*emotion:\*\*\s*([a-z]+)", text_lower)
    if match:
        val = match.group(1).strip()
        if val in EMOTION_NAMES:
            return val

    # 2. Check for "emotion is <category>" or "category is <category>"
    match = re.search(r"\b(?:emotion|category)\s+(?:is|should\s+be)\s+([a-z]+)\b", text_lower)
    if match:
        val = match.group(1).strip()
        if val in EMOTION_NAMES:
            return val

    # 3. Fallback: scan the entire text for any occurrence of valid emotion keywords working backwards
    # (Since the final answer is usually at the end of the text)
    words = re.findall(r"\b[a-z]+\b", text_lower)
    if words:
        for w in reversed(words):
            if w in EMOTION_NAMES:
                return w

    return ""


def generate_completion(model, tokenizer, text_entry: str, max_new_tokens: int = 128) -> str:
    prompt_instruction = (
        "Classify the following diary entry into one of these 6 emotions: "
        "sadness, joy, love, anger, fear, surprise.\n\n"
        f"Diary Entry: \"{text_entry}\"\n\n"
        "Provide a brief conversational explanation of your reasoning, then output your final category strictly inside the tag: **Emotion:** <category>"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_instruction}],
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HF model id to evaluate (default: {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        default=None,
        help="Path to a saved LoRA adapter dir. If set, base model is inferred from adapter_config.json.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit loading (use fp16/bf16 precision).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of test examples to evaluate (default: 50).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    target = args.adapter if args.adapter else args.model
    print(f"Loading model: {target} (load_in_4bit={args.load_in_4bit})")

    model, tokenizer = FastModel.from_pretrained(
        model_name=target,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # Set evaluation mode
    FastModel.for_inference(model)

    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="test")
    num_eval = min(args.num_examples, len(dataset))
    print(f"Evaluating first {num_eval} test examples...")

    correct = 0
    for idx in range(num_eval):
        example = dataset[idx]
        diary_text = example["text"]
        gold_label_idx = example["label"]
        gold_emotion = EMOTION_NAMES[gold_label_idx]

        completion = generate_completion(model, tokenizer, diary_text, args.max_new_tokens)
        pred_emotion = extract_emotion(completion)

        is_correct = pred_emotion == gold_emotion
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"[{idx + 1}/{num_eval}] {status}")
        print(f"  Text : \"{diary_text}\"")
        print(f"  Gold : {gold_emotion}")
        print(f"  Pred : {pred_emotion} (completion: {repr(completion)})")

    accuracy = (correct / num_eval) * 100
    print(f"\n{'=' * 40}")
    print(f"Emotion Classification Accuracy: {correct}/{num_eval} ({accuracy:.2f}%)")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
