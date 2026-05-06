# unsloth-fine-tune-gemma-4

Worked examples of [Unsloth](https://github.com/unslothai/unsloth) on Gemma 4,
starting with an inference-only smoke test for the **E2B** instruction-tuned
variant.

## Hardware

- An NVIDIA GPU with CUDA support.
- **~8 GB VRAM minimum** for the default Gemma 4 E2B model (4-bit). The
  transformers loader briefly materializes a ~4.4 GiB tensor while moving
  weights to GPU, so cards reporting 6 GB total cannot fit it.
- For smaller cards (e.g. RTX 4050 Laptop, 6 GB), pass `--model` to the
  inference script and use a smaller Gemma. Both these are tested:
  - `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (~700 MB, comfortable)
  - `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` (~2.5 GB, still fits)

## Tooling

- [`uv`](https://docs.astral.sh/uv/) for dependency / venv management.
- The `hf` CLI (ships with `huggingface_hub` ≥ 0.34, installed by `uv sync`).

## Getting started

```bash
# 1. Create the venv and install dependencies into ./.venv
uv sync

# 2. (Optional) Sign in if you plan to pull gated models from the Hub
uv run hf auth login

# 3. Pre-download the model into the shared HF cache (~/.cache/huggingface).
#    Doing this through `hf` means the same files are reusable from any other
#    tool that respects the HF cache (transformers, llama.cpp converters, etc.).
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download unsloth/gemma-4-E2B-it-unsloth-bnb-4bit

# 4. Run the inference smoke test
uv run python examples/inference.py
```

You should see a streamed answer to *"What is the capital of France?"*. Pass
`--prompt "..."` to ask something else, or `--model <hf-id>` to run a different
checkpoint.

On a small (6 GB) GPU:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download unsloth/gemma-3-1b-it-unsloth-bnb-4bit
uv run python examples/inference.py --model unsloth/gemma-3-1b-it-unsloth-bnb-4bit
```

## Fine-tuning example

`examples/finetune_gsm8k.py` is a getting-started SFT example: it runs three
held-out GSM8K problems through the model, fine-tunes a LoRA adapter on a
1000-row subset of the GSM8K train split, then runs the same problems again
so you can see the change. The headline is a format imprint — the base
model answers in prose, the fine-tuned model produces step-by-step CoT
ending with `#### <answer>`.

Defaults are sized for a 6 GB consumer GPU: Gemma 3 1B in 4-bit, QLoRA at
`max_seq_length=2048`, peak ~3 GB VRAM, ~5–10 min for the 60-step demo.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download unsloth/gemma-3-1b-it-unsloth-bnb-4bit
uv run python examples/finetune_gsm8k.py
```

The adapter is saved to `lora_gsm8k/`. Pass `--model <hf-id>` to fine-tune
a different Gemma checkpoint, `--max-steps` to train longer, or
`--train-rows 0` to use the full GSM8K train split.

To re-evaluate a saved adapter without retraining:

```bash
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k/
```

## Experiments

`examples/finetune_gsm8k.py` exposes the training-relevant knobs as CLI
flags so you can vary one or two at a time and compare. A few starter
configurations:

```bash
# More training, more data, defaults otherwise
uv run python examples/finetune_gsm8k.py \
  --max-steps 300 \
  --train-rows 2000 \
  --output-dir lora_gsm8k_exp2

# Higher LoRA rank + tuned hyperparameters
uv run python examples/finetune_gsm8k.py \
  --max-steps 500 \
  --train-rows 3000 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --learning-rate 1e-4 \
  --warmup-steps 20 \
  --lr-scheduler-type cosine \
  --output-dir lora_gsm8k_exp3
```

Each run writes its adapter to a different directory. To evaluate any
saved adapter against the same fixed eval set:

```bash
uv run python examples/eval_gsm8k.py --adapter <output-dir>
```

See [experiments.md](experiments.md) for a full description of the
experiments that have been run, results table, and what each artifact in
an output directory contains.

## Layout

```
.
├── pyproject.toml             # uv-managed dependencies
├── examples/
│   ├── inference.py           # Unsloth + Gemma inference smoke test
│   ├── finetune_gsm8k.py      # GSM8K fine-tuning with before/after eval
│   └── eval_gsm8k.py          # Greedy eval of a base model + optional LoRA adapter
├── experiments.md             # Documented experiments + reproduction commands
└── README.md
```

## Future work

### Running with larger Gemma variants

All scripts accept `--model <hf-id>`, so they already work with bigger
checkpoints when you have the VRAM. Plausible next steps:

- `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` — ~2.5 GB for inference (fits
  6 GB), ~8–10 GB for QLoRA training (needs a larger GPU). Per
  [experiments.md](experiments.md), moving to a 4B base is the most
  likely way to lift GSM8K accuracy past the 1B ceiling.
- `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` — ~5 GB for inference, ~8–10 GB
  for QLoRA. Multimodal-capable; the multimodal-list message format
  already used in `examples/inference.py` is the right shape to extend
  to image/audio inputs later.

### Conversion and deployment with LiteRT-LM

Once you have a fine-tuned adapter in `lora_gsm8k/`, the natural next
step is exporting a deployment-ready bundle:

1. Merge the LoRA adapter into the base via Unsloth's
   `model.save_pretrained_merged(...)` (writes 16-bit safetensors).
2. Convert to a `.litertlm` / `.task` bundle using the
   [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) toolchain
   (Google AI Edge's on-device LLM runtime, the LLM-focused evolution of
   LiteRT / MediaPipe LLM Inference).
3. Run on-device — phones, embedded boards, or any LiteRT-LM target.

A follow-up `examples/export_litertlm.py` is the natural place for this
flow once the upstream conversion scripts stabilize. Until then,
treat `lora_gsm8k/` as the handoff point to the LiteRT-LM tools.
