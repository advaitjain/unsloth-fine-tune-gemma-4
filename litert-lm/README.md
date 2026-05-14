# LiteRT-LM Inference for GSM8K

This folder contains the complete end-to-end pipeline for converting fine-tuned LoRA adapters into highly optimized `.litertlm` formats and evaluating them on the GSM8K mathematical reasoning task using [LiteRT-LM](https://ai.google.dev/edge/litert-lm).

---

## Prerequisites

Ensure your dependencies are fully synced via `uv`:
```bash
uv sync
```

---

## Automated Pipeline Execution

You can run the entire conversion, qualitative verification, and quantitative evaluation pipeline using our standalone bash script:

```bash
./litert-lm/convert_and_eval.sh
```

---

## Step-by-Step Manual Guide

If you prefer executing individual steps manually, follow this sequence:

### 1. Install Compilation Tools
LiteRT-LM compilation requires `litert-torch-nightly`:
```bash
uv pip install litert-lm-api
uv tool install litert-torch-nightly
```

### 2. Merge LoRA Adapter into Base Safetensors
Merge our best performing math adapter (`gsm8k-math/lora_exp17` -> **83.00% EM**) into the base model weights in full 16-bit precision:
```bash
uv run python litert-lm/merge_adapter.py \
  --adapter gsm8k-math/lora_exp17 \
  --output-dir litert-lm/merged_model
```

### 3. Export to `.litertlm` Format
Convert the merged Hugging Face safetensors into a compiled LiteRT-LM model, applying the appropriate chat template override:
```bash
mkdir -p litert-lm/compiled_model
litert-torch export_hf \
  --model=litert-lm/merged_model \
  --output_dir=litert-lm/compiled_model \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm
```

### 4. Qualitative Interactive Verification (CLI)
Run the compiled model directly from the terminal using the `litert-lm` CLI to inspect reasoning traces for specific word problems:
```bash
litert-lm run litert-lm/compiled_model/model.litertlm \
  --prompt="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
```

#### Qualitative Inference Sample (Janet's Ducks)

Executing the compiled `.litertlm` model via the `litert-lm run` CLI generates structured reasoning outputs matching the formatting integrated during supervised fine-tuning.

*   **Target Output**:
    ```
    18
    ```
*   **Compiled LiteRT-LM Output**:
    ```markdown
    Here's how to solve the problem step-by-step:

    1. **Calculate the total number of eggs Janet has:**
       Janet's ducks lay 16 eggs per day.

    2. **Calculate the total number of eggs Janet uses:**
       She eats 3 for breakfast + bakes 4 for friends = 3 + 4 = 7 eggs.

    3. **Calculate the number of eggs remaining to sell:**
       16 total eggs - 7 used eggs = 9 eggs remaining.

    4. **Calculate the total money she makes:**
       9 eggs * $2 per egg = $18

    **Answer:** Janet makes **$18** every day at the farmers' market.
    ```

#### Quantized Models Qualitative Comparison

| Model | Verbatim Reasoning Trace Output |
| :--- | :--- |
| **LiteRT-Community baseline**<br>`gemma-4-E2B-it.litertlm` | Here's how to solve the problem step-by-step:<br><br>1. **Calculate the total number of eggs Janet has:**<br>   Janet's ducks lay 16 eggs per day.<br><br>2. **Calculate the number of eggs Janet eats:**<br>   She eats 3 eggs for breakfast.<br><br>3. **Calculate the number of eggs Janet uses for muffins:**<br>   She uses 4 eggs for muffins.<br><br>4. **Calculate the total number of eggs used:**<br>   Eggs eaten + Eggs for muffins = $3 + 4 = 7$ eggs.<br><br>5. **Calculate the number of eggs remaining for sale:**<br>   Total eggs laid - Total eggs used = $16 - 7 = 9$ eggs.<br><br>6. **Calculate the total money she makes:**<br>   Number of eggs for sale $\times$ Price per egg = $9 \times \$2 = \$18$.<br><br>**Answer:** Janet makes **\$18** every day at the farmers' market. |
| **`dynamic_wi8c_afp32`** | Here's how to solve the problem step-by-step:<br><br>1. **Calculate the total number of eggs laid:**<br>   Janet's ducks lay 16 eggs per day.<br><br>2. **Calculate the total number of eggs she eats and bakes:**<br>   She eats 3 for breakfast + bakes 4 for friends = $3 + 4 = 7$ eggs.<br><br>3. **Calculate the number of remaining eggs sold:**<br>   Total eggs laid - Eggs eaten/baked = $16 - 7 = 9$ eggs.<br><br>4. **Calculate the total money she makes:**<br>   Number of eggs sold $\times$ Price per egg = $9 \times \$2 = \$18$.<br><br>**Answer:** Janet makes **$18** every day at the farmers' market. |
| **`gemma4_mixed48_hr`** | Here's how to solve the problem step-by-step:<br><br>1. **Calculate the number of eggs Janet eats:** She eats 3 eggs every morning.<br><br>2. **Calculate the number of eggs she bakes:** She bakes 4 muffins, which implies she uses 4 eggs for the muffins (or the problem intends for the muffins to be made from eggs, but the context is about selling the *remainder* of the laid eggs).<br><br>3. **Calculate the number of eggs remaining to be sold:**<br>   * Total eggs laid: 16<br>   * Eggs eaten/used: $16 - 3 - 4 = 9$ eggs remaining.<br><br>4. **Calculate the total earnings:** She sells the remaining eggs at $\$2$ each.<br>   * $9 \text{ eggs} \times \$2/\text{egg} = \$18$<br><br>**Answer:** She makes **\$18** every day at the farmers' market. |
| **`gemma4_mixed48_b32`** | Here's how to solve the problem step-by-step:<br><br>1. **Find the total number of eggs Janet has:**<br>   Janet's ducks lay 16 eggs per day.<br><br>2. **Find the number of eggs she eats:**<br>   She eats 3 eggs for breakfast.<br><br>3. **Find the number of eggs she bakes:**<br>   She bakes 4 muffins (which implies she uses 4 eggs, as the problem states she bakes muffins *with* four, but the context implies these are eggs being used).<br><br>4. **Find the number of remaining eggs:**<br>   Total eggs - Eggs eaten - Eggs baked = Remaining eggs<br>   $16 - 3 - 4 = 9$ eggs<br><br>5. **Calculate the total money she makes:**<br>   She sells the remainder at $2 per egg.<br>   $9 \text{ eggs} \times \$2/\text{egg} = \$18$<br><br>**Answer:** She makes **$18** every day at the farmers' market. |
| **`gemma4_mixed48_b64`** | Here's how to solve the problem step-by-step:<br><br>1. **Find the number of eggs Janet has left after breakfast and baking:**<br>   * She starts with 16 eggs.<br>   * She eats 3 for breakfast: $16 - 3 = 13$ eggs.<br>   * She bakes 4 muffins (This part of the problem is a bit confusing. It says "she bakes muffins for her friends every day with four." We need to assume this means she uses 4 eggs for the muffins).<br>   * Eggs remaining after baking: $13 - 4 = 9$ eggs.<br><br>2. **Calculate the total money she makes at the market:**<br>   * She sells the remainder (9 eggs) at $2 per egg.<br>   * $9 \times \$2 = \$18$<br><br>**Answer:** Janet makes **$18** every day at the farmers' market. |

### 5. Quantitative Batch Evaluation (Python API)
Evaluate the compiled model against our 100 held-out GSM8K validation problems. The evaluation script leverages `litert_lm.Engine` on a robust CPU backend to ensure execution stability across target execution environments:
```bash
uv run python litert-lm/eval_litert_gsm8k.py --model litert-lm/compiled_model/model.litertlm
```

#### Quantitative Benchmark Performance

Executing batch evaluations across the held-out 100-problem validation set on the CPU engine backend yields the following metrics:

| Model Configuration | Quant Scheme | Model Size | Score (Exact Match) | Status |
| :--- | :--- | :---: | :---: | :---: |
| Base Instruct (`unsloth/gemma-4-E2B-it`) | None (FP16 Safetensors) | 9.6 GB | **84.00%** | Baseline |
| LiteRT-Community [gemma-4-E2B-it.litertlm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/blob/main/gemma-4-E2B-it.litertlm) | `custom` | 2.58 GB | **78.00%** | Completed |
| Merged LoRA Adapter (`litert-lm/merged_model/`) | None (FP16 Safetensors) | 9.6 GB | **83.00%** | Completed |
| LiteRT-LM int8 | `dynamic_wi8c_afp32` | 5.07 GB | **83.00%** | Completed |
| LiteRT-LM mixed int8/int4+hadamard rotations | `gemma4_mixed48_hr` | 2.59 GB | **77.00%** | Completed |
| LiteRT-LM mixed int8 per-channel, blockwise (32) int4 | `gemma4_mixed48_b32` | 2.86 GB | **77.00%** | Completed |
| LiteRT-LM mixed int8 per-channel, blockwise (64) int4 | `gemma4_mixed48_b64` | 2.70 GB | **77.00%** | Completed |
