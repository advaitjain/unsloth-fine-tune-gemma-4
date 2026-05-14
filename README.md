# unsloth-fine-tune-gemma-4

Worked examples of [Unsloth](https://github.com/unslothai/unsloth) on Gemma 4 E2B.

---

## Getting Started

### Hardware Requirements

* **NVIDIA GPU with at least 16 GB of VRAM is required**.
  - **Base Inference**: 4-bit E2B requires **8.3 GB**, and 16-bit fp16 E2B requires **10.3 GB** of VRAM.
  - **SFT LoRA Training**: Standard 16-bit full-precision SFT adapter training requires **11.7 GB** of VRAM.
  - This repo has been tested and benchmarked on a single **NVIDIA GeForce RTX 4090 (24 GB VRAM)**.


### Environment Setup

We use `uv` for virtual environment and dependency management.

```bash
# 1. Create virtual environment and sync dependencies
uv sync

# 2. Authenticate Hugging Face CLI to download gated weights
uv run hf auth login

# 3. Download the E2B model directly into the shared HF cache (~/.cache/huggingface)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download unsloth/gemma-4-E2B-it-unsloth-bnb-4bit
```

### Inference Smoke Test

Ensure the base model streams completions successfully:

```bash
uv run python examples/inference.py
```
*Expectation*: Streams the answer to *"What is the capital of France?"*. Pass `--prompt "..."` or `--model <hf-id>` to customize.

---

## GSM8K SFT (Arithmetic Reasoning)

Worked examples of causal language model supervised fine-tuning (SFT) for logical reasoning and multi-step word problem solving using the canonical [GSM8K (Grade School Math 8K)](https://github.com/openai/grade-school-math) dataset hosted on [Hugging Face](https://huggingface.co/datasets/openai/gsm8k).

**LiteRT-LM Deployment**: Conversion, quantization, and evals with LiteRT-LM is a work-in-progress with details in [litert-lm/README.md](litert-lm/README.md).

We fine-tune **Gemma 4 E2B** in full **16-bit precision (FP16)**, applying a $1\times10^{-5}$ learning rate to integrate custom turn markers (`<|turn>user\n`) and structured prose while maintaining zero-shot accuracy.

```bash
# A. Train 16-bit fp16 SFT adapter (r=32, alpha=64, LR 1e-5, Cosine scheduler, 200 steps)
uv run python gsm8k-math/finetune_gsm8k.py \
  --model unsloth/gemma-4-E2B-it \
  --no-4bit \
  --lora-rank 32 \
  --lora-alpha 64 \
  --learning-rate 1e-5 \
  --lr-scheduler-type cosine \
  --train-rows 3000 \
  --max-steps 200 \
  --output-dir gsm8k-math/lora_exp17

# B. Evaluate the trained LoRA adapter offline (greedy exact match verification over held-out test set)
uv run python gsm8k-math/eval_gsm8k.py --adapter gsm8k-math/lora_exp17/ --no-4bit --eval-rows 100

# C. Run interactive demonstrator comparing Zero-Shot base outputs vs. SFT adapter side-by-side
uv run python gsm8k-math/inference_demo.py --adapter gsm8k-math/lora_exp17/
```

#### Qualitative Fine-Tuning Delta (Before vs. After SFT):

SFT successfully trains the model to structure its logical reasoning into Markdown lists and explicit arithmetic formulas, resulting in an extracted value accuracy of **83.00% Exact Match** (compared to the 84.00% zero-shot baseline).

##### 1. Structuring Multi-Step Arithmetic Formats (Janet's Ducks)

*   **Input Word Problem**:
    ```
    Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    ```
*   **EXPECTED GROUND TRUTH (GT)**:
    ```
    18
    ```
*   **BEFORE SFT (Zero-Shot Base Model)**:
    ```markdown
    Here's how to solve the problem step-by-step:

    1. **Calculate the total number of eggs laid:**
       Janet's ducks lay 16 eggs per day.

    2. **Calculate the total number of eggs she uses:**
       * Eggs for breakfast: 3
       * Eggs for muffins: 4
       * Total eggs used: $3 + 4 = 7$ eggs

    3. **Calculate the number of remaining eggs (eggs sold):**
       * Eggs remaining: $16 - 7 = 9$ eggs

    4. **Calculate the total money she makes:**
       * She sells the remainder (9 eggs) at $\$2$ per egg.
       * Earnings: $9 \times \$2 = \$18$

    **Answer:** Janet makes **\$18** every day at the farmers' market.
    ```
*   **AFTER SFT (Fine-Tuned LoRA Model)**:
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
    *Generates enumerated steps, explicit addition/multiplication bounds, and bolded answers.*

See [gsm8k-math/experiments.md](gsm8k-math/experiments.md) for the complete 18-experiment hyperparameter sweeps matrix and comparative evaluations across data volumes and learning rates.

## Multimodal LaTeX OCR SFT

Re-implementation of Unsloth's [Gemma 4 Training
Guide](https://unsloth.ai/docs/models/gemma-4/train). Additional details of
parameter sweeps that we attempted are in [this doc](latex_ocr_experiments.md).

```bash
# A. Run zero-shot base model baseline (N=50)
uv run python examples/eval_vision.py --model unsloth/gemma-4-E2B-it --no-4bit --eval-rows 50 --vision-tokens 280

# B. Train 16-bit fp16 SFT peak-LoRA adapter (r=32, alpha=64, LR 5e-5, 280 tokens)
uv run python examples/finetune_vision.py \
  --model unsloth/gemma-4-E2B-it \
  --no-4bit \
  --lora-rank 32 \
  --lora-alpha 64 \
  --learning-rate 5e-5 \
  --vision-tokens 280 \
  --max-steps 60 \
  --output-dir lora_vision_best

# C. Evaluate the trained SFT adapter
uv run python examples/eval_vision.py --model lora_vision_best/ --no-4bit --eval-rows 50 --vision-tokens 280

# D. Merge LoRA adapters back into base weights to produce a 16-bit safetensors directory
uv run python examples/merge_lora.py --adapter lora_vision_best --output-dir lora_vision_merged_fp16

# E. Run streaming formula inference on a sample local image using the merged standalone safetensors
uv run python examples/inference.py --model lora_vision_merged_fp16/
```

#### Qualitative Fine-Tuning Delta (Before vs. After SFT):

To render these LaTeX formula strings directly in your terminal local environment, run:
```bash
# Setup tool once
uv tool install texicode
# Render formula
txc "<latex_formula_string>"
```

##### 1. Visual Symbol Resolution (Alpha vs. English 'a')

*   **Input Image**:
    ![sample_1.png](examples/sample_1.png)
*   **EXPECTED GROUND TRUTH (GT)**:
    ```latex
    \omega _ { a b } ^ { \alpha \beta } ( x , y ) = m ^ { 2 } \epsilon ^ { \alpha \beta } \delta ^ { a b } \delta ( x - y )
    ```
*   **BEFORE SFT (Zero-Shot Base Model)**:
    ```latex
    \omega_{ab}^{a\beta}(x,y) = m^2 \epsilon^{a\beta} \delta^{ab} \delta(x-y)
    ```
*   **AFTER SFT (Fine-Tuned Peak LoRA Model)**:
    ```latex
    \omega_{ab}^{\alpha\beta}(x, y) = m^2 \epsilon^{\alpha\beta} \delta^{ab} \delta(x - y)
    ```
    *Uses mathematical Greek \alpha letter instead of a.*

##### 2. Detail Omission Prevention (Compose Circle `\circ`)

*   **Input Image**:
    ![sample_2.png](examples/sample_2.png)
*   **EXPECTED GROUND TRUTH (GT)**:
    ```latex
    \nabla _ { \mu } = T \circ \partial _ { \mu } \circ T ^ { + } + \Pi \circ \partial _ { \mu } \circ \Pi + \rho _ { \mu }
    ```
*   **BEFORE SFT (Zero-Shot Base Model)**:
    ```latex
    \nabla_{\mu} = T \circ \partial_{\mu} T^{+} + \Pi \circ \partial_{\mu} \Pi + \rho_{\mu}
    ```
*   **AFTER SFT (Fine-Tuned Peak LoRA Model)**:
    ```latex
    \text{\nabla}_{\mu} = T \circ \partial_{\mu} \circ T^{+} + \Pi \circ \partial_{\mu} \circ \Pi + \rho_{\mu}
    ```
    *Preserves mathematical compose circle \circ operators.*

##### 3. Mathematical Syntax Alignment (`\cdots` vs. `\dots`)

*   **Input Image**:
    ![sample_3.png](examples/sample_3.png)
*   **EXPECTED GROUND TRUTH (GT)**:
    ```latex
    n _ { i } = m _ { i } + m _ { i + 1 } + \cdots + m _ { N - 1 } + n _ { N } .
    ```
*   **BEFORE SFT (Zero-Shot Base Model)**:
    ```latex
    n_i = m_i + m_{i+1} + \dots + m_{N-1} + n_N.
    ```
*   **AFTER SFT (Fine-Tuned Peak LoRA Model)**:
    ```latex
    n_i = m_i + m_{i+1} + \cdots + m_{N-1} + n_N.
    ```
    *Replaces common text lower ellipsis with math centered dots \cdots.*



## CUAD Legal SFT (Contract Understanding)

Worked examples of parameter-efficient fine-tuning (PEFT) on legal commercial contracts for key-value clause extraction using the expert-annotated [CUAD (Contract Understanding Atticus Dataset)](https://github.com/TheAtticusProject/cuad) hosted on [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad).

We fine-tune **Gemma 4 E2B** in full **16-bit precision (FP16)** to extract the exact contract segment specifying the `Governing Law` clause.

```bash
# A. Train 16-bit fp16 SFT Peak-LoRA adapter (r=32, alpha=64, LR 1e-4, Cosine scheduler, 160 steps)
uv run python cuad/finetune_cuad.py \
  --model unsloth/gemma-4-E2B-it \
  --no-4bit \
  --lora-rank 32 \
  --lora-alpha 64 \
  --learning-rate 1e-4 \
  --lr-scheduler-type cosine \
  --max-steps 160 \
  --output-dir lora_cuad_best

# B. Evaluate the trained LoRA SFT adapter offline (greedy EM and F1 average)
uv run python cuad/eval_cuad.py --adapter lora_cuad_best/ --no-4bit --eval-rows 50

# C. Run interactive demonstrator comparing Zero-Shot base outputs vs. SFT adapters side-by-side
uv run python cuad/inference_demo.py --adapter lora_cuad_best/
```

#### Qualitative Fine-Tuning Delta (Before vs. After SFT):

SFT successfully trims surrounding text and heading numbers noise from raw generations, achieving a **+20.00% absolute Exact Match (EM) average improvement** over Zero-Shot baselines (EM metrics jump from `62.00%` to **`82.00%`**).

##### 1. Removing Section Numbering and Structural Headings Noise

*   **Input Context Snippet**:
    ```
    ensor hereunder, which is properly payable by Customer, and after Customer has m
    et withholding requirements, Customer shall pay to Licensor on demand the full a
    mount of such additional withholding or intercepted payment.

    17. GENERAL

    17.1. Governing Law. The validity, construction and interpretatio...
    ```
*   **EXPECTED GROUND TRUTH (GT)**:
    ```
    The validity, construction and interpretation of this Agreement and the rights and duties of the parties hereto shall be governed by the internal laws of the State of New York, excluding its principles of conflict of laws.
    ```
*   **BEFORE SFT (Zero-Shot Base Model)**:
    ```
    17.1. Governing Law. The validity, construction and interpretation of this Agreement and the rights and duties of the parties hereto shall be governed by the internal laws of the State of New York, excluding its principles of conflict of laws.
    ```
    *Fails exact match bounds because the base instruct model pulls in the surrounding section heading text "17.1. Governing Law.".*
*   **AFTER SFT (Fine-Tuned LoRA Model)**:
    ```
    The validity, construction and interpretation of this Agreement and the rights and duties of the parties hereto shall be governed by the internal laws of the State of New York, excluding its principles of conflict of laws.
    ```
    *Perfectly aligns and trims segment bounds, matching the exact annotated substring.*

See [cuad/experiments.md](cuad/experiments.md) for the complete 10-config hyperparameter sweeps matrix, representation capacity analysis, and additional SFT takeaways.


## Other Experiments


### Some Takeaways

Across three distinct text-only tasks (GSM8K math, Regex semantic parsing, and Emotion classification), **SFT on highly capable instruction-tuned LLMs (Gemma 4 E2B) consistently underperforms Zero-Shot baseline prompting**. We observed three distinct limiting dynamics:

1. **The Generalization Bottleneck (NL-to-Regex)**:
   - The base model scores **2.00%** exact match (EM) accuracy zero-shot because it generates standard PCRE regexes (like `^[^e]*$`), while the dataset targets a custom logical **LRegex** dialect (`~(.*e.*)`).
   - SFT training successfully overfit the training split (collapsing SFT loss to $0.198$). However, when evaluated on held-out test prompts, the SFT model scored **0.00%** accuracy.
   - SFT on a small dataset (724 rows) acts as a memorization cache; the QLoRA adapter lacks the semantic capacity to synthesize and generalize a completely new logical grammar syntax to unseen prompts. Consequently, the base model's massive pre-trained PCRE regex prior dominates.
   - See [experimental/README.md](experimental/README.md) for the Regex study.

2. **Human Annotator Benchmark Noise (Emotion Classification)**:
   - The base model achieved a baseline of **54.00%** accuracy on the `dair-ai/emotion` dataset, which shifted slightly to **56.00%** after SFT.
   - Manual trace audits revealed that the model's "mismatches" were actually highly logical, conceptually correct classifications (e.g., mapping `"i feel so cold"` to `sadness` instead of the gold label `anger`, or `"friendly affection"` to `love` instead of the gold label `joy`).
   - Because E2B maintains a rigid pre-trained semantic logic under gentle SFT, it refuses to overfit to noisy, inconsistent human annotations, limiting its exact-match accuracy improvement.
   - See [experimental/README.md](experimental/README.md) for the Emotion study.


---

## Repository Structure

```
.
├── pyproject.toml                  # uv dependency definitions
├── experiments.md                  # Structured GSM8K results, findings & sweeps table
├── latex_ocr_experiments.md        # Complete Master 50-run sweeps and 1-Epoch results for LaTeX OCR
├── README.md                       # Upfront takeaways, hardware bounds & reproduction index
├── latex_ocr/                      # Sequential sweeps coordinator and researcher tools package
│   ├── run_master_sweeps.py        # Sequential sweeps trainer + evaluator
│   ├── run_full_epoch_sequel.py    # sequel scaled full epoch tuner
│   └── inspect_processor.py        # processor structures and token budgets analyzer
├── cuad/                           # Legal contract extraction SFT researchers namespace
│   ├── finetune_cuad.py            # SFT training loops, boundary offset and validation evaluator
│   ├── eval_cuad.py                # stand-alone greedy metrics EM/F1 validation solver
│   ├── inference_demo.py           # comparative Zero-shot vs. SFT 5-contracts visual reporter CLI
│   ├── experiments.md              # complete 10-config parameter sweeps audit table & insights
│   └── implementation_plan.md      # parameters configuration proposal plan
├── gsm8k-math/                     # Arithmetic reasoning SFT researchers namespace
│   ├── finetune_gsm8k.py           # SFT training loops, masked turn tokens and regex evaluator
│   ├── eval_gsm8k.py               # stand-alone greedy exact match validation solver
│   ├── inference_demo.py           # comparative Zero-shot vs. SFT 5-problems visual reporter CLI
│   ├── experiments.md              # complete 18-experiment parameter sweeps audit table & insights
│   └── implementation_plan.md      # parameters configuration proposal plan
├── litert-lm/                      # LiteRT-LM conversion, quantization, and evaluation pipeline
│   ├── README.md                   # Step-by-step manual guide and benchmark performance
│   ├── convert_and_eval.sh         # Automated end-to-end conversion and evaluation script
│   ├── merge_adapter.py            # Merges LoRA adapter to full precision base weights
│   └── eval_litert_gsm8k.py        # Batch evaluation script for compiled .litertlm models
├── examples/
│   ├── inference.py                # Base model and merged safetensors formula inference text
│   ├── finetune_gsm8k.py           # GSM8K QLoRA and fp16 trainer
│   ├── finetune_vision.py          # LaTeX OCR multimodal fp16 trainer (epochs-limit compatible)
│   ├── eval_vision.py              # High-precision greedy EM/NED validation evaluator (N=50)
│   ├── merge_lora.py               # high-precision fp16 LoRA merging standalone safetensors tool
│   ├── sample_1.png                # visual Greek alpha superscript target image
│   ├── sample_2.png                # visual compose circle omission target image
│   ├── sample_3.png                # visual centered dots token alignment target image
│   ├── eval_gsm8k_automated.py     # Regex-based automated parser v3 evaluator (N=50)
│   └── eval_gsm8k.py               # original manual 3-example validation helper
└── experimental/
    ├── README.md                   # Unified design, findings and tables for Regex & Emotion tasks
    ├── train_regex.py              # High-LR QLoRA regex trainer
    ├── eval_regex.py               # Markdown-extracting regex EM evaluator
    ├── train_emotion.py            # QLoRA SFT trainer with response masking
    └── eval_emotion.py             # Multi-tier fallback keyword emotion evaluator
```
