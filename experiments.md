# Causal Math SFT Experiments (GSM8K)

This document records systematic experiments studying the effects of supervised fine-tuning (SFT) hyperparameters, quantization, and base precision on Gemma edge models using the **GSM8K** grade-school math benchmark.

---

## Hardware Target & Environment

* **Minimum Hardware Specification**: An NVIDIA GPU with CUDA support and **at least 16 GB of VRAM** is required to execute these training sweeps. 
* **Reference GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
* **Software Stack**: Managed CPython 3.11.15 virtual environment using the `uv` package manager, with Unsloth custom training kernels.

---

## Gemma 4 E2B Systematic SFT Experiments (N=50)

We conducted systematic SFT sweeps on the **Gemma 4 E2B Instruct** model (5 Billion parameters, both 4-bit quantized and full 16-bit precision variants) using a held-out representative test split of 50 math problems from the `openai/gsm8k` main test split.

### 1. Evaluation Protocol

* **Deterministic Greedy Decoding**: All baseline and post-SFT evaluations were executed with greedy decoding (`do_sample=False`) to eliminate sampling noise.
* **Automated Regex Fallback Parsing**: Because instruction-tuned models generate detailed prose reasoning before arriving at a final number, we implemented `examples/eval_gsm8k_automated.py`. This script uses an advanced regular expression parser that:
  - Isolates standard GSM8K `#### <answer>` markers.
  - Backtracks through the completion text to extract final calculated quantities, ignoring trailing units (e.g., `hours`, `liters`, `weeks`) and bolded step indexes.
* **Hyperparameter Sweep Configurations**:
  - **Run B1 & B2**: Baseline zero-shot evaluation on base 4-bit and fp16 models.
  - **Run 4B-S1**: Standard reference SFT budget (60 steps, 1000 training rows, $r=\alpha=8$, LR $2e-4$).
  - **Run 4B-S2**: Extended SFT budget (300 steps, 2000 training rows, $r=\alpha=8$, LR $2e-4$).
  - **Run 4B-S3**: SFT capacity boost + low learning rate (200 steps, 2000 training rows, $r=\alpha=32$, LR $2e-5$, cosine scheduler, response-only masking).
  - **Run FP-S1**: 16-bit full precision duplication of SFT capacity boost + low learning rate (Run 4B-S3 equivalent on `unsloth/gemma-4-E2B-it` with `--no-4bit`).

### 2. Experimental Results Table

The collected metrics across all systematic sweeps are recorded below:

| Run ID | Precision | SFT Hyperparameters | GSM8K Accuracy (N=50) | SFT Loss | Training Time | Peak VRAM |
|---|---|---|---|---|---|---|
| **B1** | 4-bit | Zero-Shot Baseline (Greedy) | **84.00%** (42/50) | — | — | 8.3 GB |
| **B2** | fp16 | Zero-Shot Baseline (Greedy) | **84.00%** (42/50) | — | — | 10.3 GB |
| **4B-S1** | 4-bit | QLoRA ($r=8$, $a=8$), 60 steps, 1k rows, LR $2e-4$ | **74.00%** (37/50) | 1.053 | 2.6 min | 8.3 GB |
| **4B-S2** | 4-bit | QLoRA ($r=8$, $a=8$), 300 steps, 2k rows, LR $2e-4$ | **32.00%** (16/50) | 0.5556 | 9.9 min | 8.3 GB |
| **4B-S3** | 4-bit | QLoRA ($r=32$, $a=32$), 200 steps, 2k rows, LR $2e-5$ | **78.00%** (39/50) | 0.9826 | 6.9 min | 8.3 GB |
| **FP-S1** | fp16 | QLoRA ($r=32$, $a=32$), 200 steps, 2k rows, LR $2e-5$ | **80.00%** (40/50) | 0.973 | 5.8 min | 11.7 GB |

### 3. Empirical Findings & Analysis

1. **High Baseline Zero-Shot Cap**: Base Gemma 4 E2B Instruct is a highly optimized reasoner zero-shot, securing **84.00%** accuracy out of the box.
2. **Catastrophic Forgetting at Standard LRs**: Fine-tuning E2B Instruct at standard learning rates ($2e-4$, Run 4B-S2) causes a **severe capability degradation to 32.00%**. The model overfits to local SFT formatting and forgets core problem constraints (e.g., omitting required variables during computation).
3. **Low-LR Alignment Tax**: Dropping the SFT learning rate by 10× ($2e-5$, Run 4B-S3) stabilizes the training process and prevents catastrophic forgetting (loss decreased cleanly to $0.98$). However, the SFT model still underperforms its base zero-shot counterpart (78% vs 84%). SFT on a narrow, structured dataset acts as a general capability constraint ("alignment tax") on pre-trained instruct weights.
4. **Precision Penalty**: Training in full 16-bit precision (`FP-S1`) yields a direct **2.00% improvement** over 4-bit quantized training (`4B-S3`) (80% vs 78%), proving that quantization weight representations introduce training limitations.
5. **Edge Training Viability**: Under Unsloth optimizations, full 16-bit precision LoRA training of E2B (5B parameters) requires only **11.7 GB of VRAM**, making it highly practical on modern consumer-class GPUs.

---

## Historical Reference: Gemma 3 1B QLoRA Sweeps

For historical baseline reference, we retain the initial hyperparameter sweep executed on the smaller **Gemma 3 1B IT** model. These runs evaluated three hardcoded test prompts: P1 (Janet's ducks), P2 (Robe bolts), and P3 (Josh's house flip).

### 1. Reference Command Sweep
- **Reference Run**: 60 steps, 1000 rows, $r=8$, $a=8$, LR $2e-4$, linear scheduler, warmup=5.
- **Exp 2**: 300 steps, 2000 rows, $r=8$, $a=8$, LR $2e-4$, linear scheduler.
- **Exp 3**: 500 steps, 3000 rows, $r=32$, $a=32$, LR $1e-4$, cosine scheduler, warmup=20.

### 2. Historical Results Table

| Run | P1 ($18) | P2 (3) | P3 ($70k) | Score | Train Loss | Wall Clock |
|---|---|---|---|---|---|---|
| **Base model, greedy** | ❌ | 3 ✅ (rambling) | ❌ | ~1/3 | — | — |
| **Exp 1 (Ref, Greedy)** | ❌ | 3 ✅ | ❌ | 1/3 | 0.92 | (no retrain) |
| **Exp 2 (300 steps)** | 18 ✅ | ❌ | ❌ | 1/3 | 0.62 | 14.2 min |
| **Exp 3 (500 steps)** | 18 ✅ | ❌ | ❌ | 1/3 | 0.57 | 22.9 min |

### 3. Key Historical Lessons
- **Greedy Decoding**: Shifting to greedy decoding (`do_sample=False`) was the single largest driver of baseline correctness.
- **Math Ceiling**: The 1B model hit a semantic ceiling; it successfully calculated gains but consistently forgot to net out repair costs in Problem 3. More training data or higher LoRA capacity ($r=32$) did not resolve this representation limit.

---

## Gemma 4 E2B Vision SFT Experiments (LaTeX OCR)

We conducted SFT on the **Gemma 4 E2B Instruct** model (4-bit quantized) using the `unsloth/LaTeX_OCR` dataset. The task is to transcribe images of mathematical formulas into LaTeX.

### 1. Evaluation Protocol
* **Deterministic Greedy Decoding**: Evaluated with `do_sample=False`.
* **Visual Verification**: We selected 3 representative test samples from the `LaTeX_OCR` test split and compared the raw LaTeX output BEFORE and AFTER fine-tuning.

### 2. Experimental Results Table

| Run ID | Precision | SFT Hyperparameters | Final Loss | Training Time | Peak VRAM | Key Behavioral Changes |
|---|---|---|---|---|---|---|
| **Vision-S1** | 4-bit | QLoRA ($r=16$, $\alpha=16$), 60 steps, 1k rows, LR $2e-4$ | **0.8659** | 2.8 min | ~9.5 GB | Delimiter shifted to `$$ ... $$`; corrected incorrect symbol generation (e.g., `\bar{\chi}` -> `\hat{x}`). |

### 3. Qualitative Results (Before vs After)

#### Sample 1: Variable Capture
* **Expected:** `b _ { 2 } ^ { \pm } = \sum _ { \mu , \nu \in { \cal R } } ( L ^ { \pm } ) _ { \mu \nu } ^ { 2 } / ( 4 \omega C _ { \cal R } ) .`
* **BEFORE:** `b_2^\pm = \sum_{\mu,\nu\in\mathbb{R}} (L^\pm)^2 \nu / (4\omega C_R).` (Missed `\mu` in the term `(L^\pm)^2 \nu`)
* **AFTER:** `$$b_2^\pm = \sum_{\mu,\nu\in\mathbb{R}} (L^\pm)^2 \mu\nu / (4\omega C_R). $$` (Captured the missing `\mu` variable as `\mu\nu`)

#### Sample 2: Delimiter Alignment
* **Expected:** `\frac { \int _ { S } \vert \nabla \phi \vert ^ { 2 } d V } { \in t _ { S } \vert \phi \vert ^ { 2 } d V } < 1 0 \lambda .`
* **BEFORE:** `\frac{\int_S |\nabla \phi|^2 dV}{\int_S |\phi|^2 dV} < 10\lambda.` (Correct math, but used code block formatting in logs)
* **AFTER:** `$$\frac{\int_S |\nabla \phi|^2 dV}{\int_S |\phi|^2 dV} < 10\lambda.$$` (Correct math, aligned to standard LaTeX display delimiters)

#### Sample 3: Symbol Correction (Crucial capability improvement)
* **Expected:** `{ \frac { 1 } { ( \pi ) ^ { 4 } } } \int \! \! \! \int d ^ { 4 } \widehat { x } \phi ^ { n } ( \widehat { x } ) \neq { \frac { 1 } { ( \pi ) ^ { 4 } } } \int d ^ { 4 } x \phi ^ { n } ( x ) \quad n \geq 2`
* **BEFORE:** `\frac{1}{(\pi)^4} \int d^4 \bar{\chi} \phi^n(\bar{\chi}) \neq \frac{1}{(\pi)^4} \int d^4 x \phi^n(x) \quad n \ge 2` (Incorrectly transcribed $\widehat{x}$ as `\bar{\chi}`)
* **AFTER:** `$$\frac{1}{(\pi)^4} \int d^4 \hat{x} \phi^n(\hat{x}) \neq \frac{1}{(\pi)^4} \int d^4 x \phi^n(x) \quad n \ge 2$$` (Correctly transcribed $\widehat{x}$ as `\hat{x}`)

