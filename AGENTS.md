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

### Codebase Map
*   **Text SFT & Math Reasoning**: [examples/finetune_gsm8k.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/finetune_gsm8k.py) (runs structured math tuning templates), [examples/eval_gsm8k.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/eval_gsm8k.py) (greedy decoding adapter test runner), [examples/eval_gsm8k_automated.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/eval_gsm8k_automated.py) (automated evaluation over custom splits).
*   **Vision SFT (LaTeX OCR)**: [examples/finetune_vision.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/finetune_vision.py) (VLM supervised fine-tuning loop), [examples/eval_vision.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/eval_vision.py) (Exact Match and Normalized Edit Distance vision verification pipeline).
*   **Compiled Edge Execution (LiteRT-LM)**: [examples/litert_lm_inference.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/litert_lm_inference.py) (runs compiled `.litertlm` models with CPU/GPU dynamic fallbacks).
*   **Weight Merging**: [examples/merge_lora.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/merge_lora.py) (merges adapters to full 16-bit precision safetensors).
*   **Experimental Sandbox**: [experimental/README.md](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/experimental/README.md) (task studies: mapping [experimental/train_regex.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/experimental/train_regex.py) for custom formats and [experimental/train_emotion.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/experimental/train_emotion.py) for boundary semantic matching studies).

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

Look at [examples/inference.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/inference.py) and [examples/finetune_gsm8k.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/finetune_gsm8k.py) for the canonical shape. In particular:

- Module-level docstring describing what the script does and any non-obvious
  CLI usage. Keep it short — a few lines, not a tutorial.
- Constants like `DEFAULT_MODEL`, `DATASET_NAME` near the top of the file.
- `argparse` for CLI args with `type=int|float`, `default=…`, `help="…"`.
  Use both short and long flags only where the existing scripts do
  (`-m/--model`, `-a/--adapter`, `-p/--prompt`).
- `def main() -> None:` plus `if __name__ == "__main__": main()`.
- Type hints on function signatures and helper return types.
- **Model Loading Conventions**:
  - Use generic `unsloth.FastModel` for standard loading and merging scripts, as it covers both text and vision models natively.
  - Use `unsloth.FastVisionModel` for vision training entry-points. Avoid using standard transformers loaders to guarantee Unsloth optimizations.
- **Multimodal / Vision Configuration**:
  - The target visual sequence defaults to 280 soft tokens.
  - To customize soft visual budgets dynamically, you must implement synchronization overrides directly on both the Model and Processor config objects:
    ```python
    # 1. Modify Model Config
    model.config.vision_soft_tokens_per_image = args.vision_tokens
    model.config.vision_config.default_output_length = args.vision_tokens
    # 2. Modify Processor Config
    processor.image_processor.image_seq_length = args.vision_tokens
    processor.image_processor.max_soft_tokens = args.vision_tokens
    ```
- **Chat Templates & Turn Marker Alignment**:
  - Chat messages utilize the multimodal list structure:
    ```python
    [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    ```
  - Role configurations use string `"model"`, **never** `"assistant"`.
  - Gemma 4 turn templates utilize `<|turn>user\n` and `<|turn>model\n`.
  - Gemma 3 utilizes standard `<start_of_turn>user\n` and `<start_of_turn>model\n`.
  - Ensure correct response masking triggers with `train_on_responses_only`:
    ```python
    if "gemma-4" in model_name.lower():
        instruction_part = "<|turn>user\n"
        response_part = "<|turn>model\n"
    else:
        instruction_part = "<start_of_turn>user\n"
        response_part = "<start_of_turn>model\n"
    ```
- **LiteRT-LM Compiled Execution**:
  - Load pre-compiled `.litertlm` models via `litert_lm.Engine`.
  - Handle GPU vision falls context safely to guard execution against FFI initialization failures:
    ```python
    try:
        engine = litert_lm.Engine(model_path=path, backend=litert_lm.Backend.CPU, vision_backend=litert_lm.Backend.GPU)
    except RuntimeError:
        engine = litert_lm.Engine(model_path=path, backend=litert_lm.Backend.CPU, vision_backend=litert_lm.Backend.CPU)
    ```
  - Standard terminal math renders via `texicode.pipeline.render_tex` using raw context:
    ```python
    import texicode.pipeline as tp
    rendered = tp.render_tex(latex_string, False, True, "raw", {"fonts": "normal"})
    ```
- Sampling protocols: Default sampling for chat loops recommendation is `temperature=1.0, top_p=0.95, top_k=64`. Logical reasoning reasoning and structured OCR mathematical evaluations require deterministic greedy settings: `do_sample=False`.
- Comments only when the *why* is non-obvious. Don't narrate the obvious.
- Don't add docs files, READMEs in subdirectories, or planning notes.

## Quantitative Evaluation & Normalization

To calculate precise comparative scores, predictions and gold strings must run through strict normalization procedures before validation tests are checked. Baseline evaluations use:

1.  **Regex-Based Math isolations**: Extract exact target calculations under robust patterns. Isolate specific `#### <answer>` formats, trailing measurements (e.g. weeks, hours, boxes), and format decimal conversions cleanly to avoid exact match rejection due to decimal notation style disparities (refer to `examples/eval_gsm8k_automated.py:extract_answer`).
2.  **Visual LaTeX OCR Normalizations**: Space formatting characters and common commands must be standardized:
    - Strip all white-spaces.
    - Map shortcuts cleanly: `\le(?!q)` -> `\leq`, `\ge(?!q)` -> `\geq`, `\to` -> `\rightarrow`, `\epsilon` -> `\varepsilon`.
    - Standardize braces subscript notation mappings: `_([a-zA-Z0-9]|\\[a-zA-Z]+)` -> `_{\1}` (e.g., `x_i` to `x_{i}`).
    - Validate transcription consistency mathematically using **Exact Match (EM)** and **Normalized Edit Distance (NED)** (based on customized Levenshtein calculations) in [examples/eval_vision.py](file:///usr/local/google/home/advaitjain/github/unsloth-fine-tune-gemma-4/examples/eval_vision.py).

## Experimental Insights & Hyperparameter Rules

Review validated learning takeaways from past task studies in `experiments.md` and `experimental/README.md` before planning updates:

- **The Alignment Tax constraint**: Applying training scripts over restrictive domains reduces baseline zero-shot capability parameters across overall cognitive outputs. Keep learning targets focused and verify scores relative to zero-shot baselines.
- **The Generalization Bottleneck**: Model configurations under parameter limits (e.g., 1B base scales, rank $r=8$ adapters) lack capacity elements to formulate customized, brand new semantic syntaxes (such as LRegex logic structures). In these settings, training limits cause overfitting over prompt variations, while baseline English PCRE pre-trained priors dominate evaluation queries. Do not request small QLoRA updates for large architectural/syntactic structural transitions.
- **Low-LR boundaries protecting reasoners**: When running semantic categorization tasks over noisy data categories, utilize low learning rates (such as $2e-5$) rather than standard settings ($2e-4$). This shields base weights from incorrect target designations while stabilizing capability boundaries (see `experimental/README.md` scientific audit).

## How to run things

```bash
# One-shot inference smoke test (default Gemma 4 E2B; pass --model for 1B/4B)
uv run python examples/inference.py

# GSM8K fine-tune end-to-end (BEFORE eval → train → AFTER eval → save adapter)
uv run python examples/finetune_gsm8k.py

# Vision SFT fine-tune over LaTeX OCR (enforces default 2x Alpha rule for rank)
uv run python examples/finetune_vision.py --lora-rank 16 --vision-tokens 280 --output-dir lora_vision

# Greedy eval of a saved adapter
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k/

# VLM exact score evaluation (exact score comparisons across N samples)
uv run python examples/eval_vision.py --model lora_vision/ --eval-rows 50
```

New execution variants and hyperparameter tuning operations should expose CLI commands rather than hardcoding variables into scripts.

## Verifying changes

There is no test suite. Verify in this order:

1.  **Parse check** (no GPU needed):
    ```bash
    uv run python -c "import ast; ast.parse(open('examples/<file>.py').read()); print('OK')"
    ```
2.  **End-to-end smoke run** with reduced settings:
    ```bash
    uv run python examples/finetune_gsm8k.py --max-steps 10 --train-rows 100 \
      --output-dir /tmp/lora_smoke
    ```
    Expected: BEFORE eval prints, training loss decreases, AFTER eval prints,
    adapter saved. Total ~2 min on a 6 GB GPU.
3.  **Reload check**: `uv run python examples/eval_gsm8k.py --adapter /tmp/lora_smoke`.

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
- `experimental/README.md` — documentation and reproducing guides on custom parser sandbox models.
