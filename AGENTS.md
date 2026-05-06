# AGENTS.md

Guidance for AI coding agents (Claude, Gemini, Cursor, etc.) working in
this repo. Read this before making changes.

## What this repo is

Worked examples of [Unsloth](https://github.com/unslothai/unsloth) for
inference and SFT on Gemma models. Sized end-to-end for a **6 GB consumer
GPU** (RTX 4050 Laptop class). The reader's experience is the product —
keep examples small, runnable, and explicit about VRAM costs.

User-facing docs live in `README.md`. Experiment design and historical
results live in `experiments.md`. This file is for agents.

## Hard constraint: 6 GB VRAM

Every default in the repo is chosen to fit. Before adding or changing
anything, verify:

- 4-bit (`load_in_4bit=True`, `dtype=None`).
- For QLoRA, the base model fits with training overhead — currently
  `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (~3 GB peak training).
- For inference-only, up to `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`
  (~5 GB) is OK but **its QLoRA does NOT fit** on 6 GB (~8–10 GB needed).
- Prefer the pre-quantized `-unsloth-bnb-4bit` variants — they avoid the
  transient ~4.4 GiB tensor that the regular loader materializes during
  weight transfer (this is the specific reason 6 GB cards can't load the
  non-pre-quantized E2B).
- Don't propose Gemma 4 E2B fine-tuning, full fine-tuning, or anything
  needing >6 GB without flagging the constraint and asking.

## Tooling

- **`uv`** is the package manager. Always run Python via `uv run python …`.
  Never `pip install` directly. Never invent a `requirements.txt`.
- Dependencies live in `pyproject.toml`. `unsloth` transitively pulls in
  torch, transformers, peft, trl, bitsandbytes, accelerate, xformers,
  triton — don't re-pin those unless you've hit a real conflict.
- The `hf` CLI ships with `huggingface_hub` ≥ 0.34. Pre-download models with:
  ```bash
  HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download <hf-id>
  ```
  Same HF cache (`~/.cache/huggingface`) is reused by all tools.

## Code conventions (match existing style)

Look at `examples/inference.py` and `examples/finetune_gsm8k.py` for the
canonical shape. In particular:

- Module-level docstring describing what the script does and any non-obvious
  CLI usage. Keep it short — a few lines, not a tutorial.
- Constants like `DEFAULT_MODEL`, `DATASET_NAME` near the top of the file.
- `argparse` for CLI args with `type=int|float`, `default=…`, `help="…"`.
  Use both short and long flags only where the existing scripts do
  (`-m/--model`, `-a/--adapter`, `-p/--prompt`).
- `def main() -> None:` plus `if __name__ == "__main__": main()`.
- Type hints on function signatures and helper return types.
- Use `unsloth.FastModel` (not `FastLanguageModel`) for loading. It works
  for both text-only Gemma 3 1B and multimodal Gemma 4. Don't switch
  loaders without a reason.
- Chat messages use the multimodal-list content format:
  ```python
  [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
  ```
  This works across all the Gemma variants in this repo. Plain string
  `content` works for text-only Gemma 3 too but is inconsistent with the
  rest of the codebase.
- Gemma's chat template uses role `"model"`, **not** `"assistant"`.
  `<start_of_turn>user / model` are the turn markers used by
  `train_on_responses_only`.
- Default sampling for chat: `temperature=1.0, top_p=0.95, top_k=64` (the
  Gemma team's recommendation). Default sampling for **reasoning / math
  evals**: greedy (`do_sample=False`). Don't mix these up.
- Comments only when the *why* is non-obvious. Don't narrate what the code
  is doing line-by-line.
- Don't add docs files, READMEs in subdirectories, or planning notes
  unless the user asks.

## How to run things

```bash
# One-shot inference smoke test (default Gemma 4 E2B; pass --model for 1B/4B)
uv run python examples/inference.py

# GSM8K fine-tune end-to-end (BEFORE eval → train → AFTER eval → save adapter)
uv run python examples/finetune_gsm8k.py

# Greedy eval of a saved adapter
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k/
```

`finetune_gsm8k.py` exposes all training knobs as CLI flags
(`--max-steps`, `--train-rows`, `--lora-rank`, `--lora-alpha`,
`--learning-rate`, `--warmup-steps`, `--lr-scheduler-type`,
`--output-dir`). New experiments should be runnable from the command
line — don't hard-code variants in the script.

## Verifying changes

There is no test suite. Verify in this order:

1. **Parse check** (no GPU needed):
   ```bash
   uv run python -c "import ast; ast.parse(open('examples/<file>.py').read()); print('OK')"
   ```
2. **End-to-end smoke run** with reduced settings:
   ```bash
   uv run python examples/finetune_gsm8k.py --max-steps 10 --train-rows 100 \
     --output-dir /tmp/lora_smoke
   ```
   Expected: BEFORE eval prints, training loss decreases, AFTER eval prints,
   adapter saved. Total ~2 min on a 6 GB GPU.
3. **Reload check**: `uv run python examples/eval_gsm8k.py --adapter /tmp/lora_smoke`.

For long training runs, launch via `Bash` with `run_in_background=true` and
wait for the completion notification. Don't poll. Don't run a parallel
`Monitor` task — there's an observed (not fully diagnosed) cascade where
killing a monitor can take down the training process. If you need progress
updates, just read the log file periodically.

## Training artifacts

After a training run, `--output-dir` contains `adapter_config.json`,
`adapter_model.safetensors`, the saved tokenizer, an auto-generated
`README.md`, and `checkpoint-N/`. See `experiments.md` for what each file
means.

`lora_*/` is `.gitignore`d. **Never commit training artifacts.** Stage
files explicitly with `git add <file>`; never `git add -A`/`.`.

## Don'ts (caught the hard way)

- Don't use `temperature=1.0` to evaluate math/reasoning. Sampling noise
  dominates whatever the model actually learned. Greedy decoding alone
  took the GSM8K eval from 0/3 to 1/3 on the same adapter.
- Don't truncate eval generation at 256 tokens — GSM8K reasoning chains
  exceed that. The default in `generate()` is 512 for a reason.
- Don't recommend bigger LoRA rank or longer training as the first move
  to improve accuracy on a small base model. At Gemma 3 1B scale, the
  base capability is the ceiling — see `experiments.md`.
- Don't commit `lora_*/` directories or any large weights. They're build
  output, not source.
- Don't bypass `uv` (no system pip, no `python …` directly).
- Don't add `--no-verify`, `--force`, or skip-hooks flags to git unless
  the user asks. Same for any destructive git operation.

## Where to find more

- `README.md` — user-facing setup and getting-started.
- `experiments.md` — documented hyperparameter sweeps + results table +
  artifact glossary + full CLI flag reference.
- `pyproject.toml` — dependency list and Python version range.
- `examples/inference.py` — minimal `FastModel` reference; copy its style.
