# Gemma 4 E2B Vision SFT Experiments (LaTeX OCR)

This document outlines the systematic experimental design to study the effects of vision layer fine-tuning, LoRA capacity, learning rates, and training duration on the **Gemma 4 E2B** multimodal model for the **LaTeX OCR** mathematical transcription task.

All experiments are executed in full **16-bit precision (fp16)** without QLoRA, targeting the NVIDIA GeForce RTX 4090 (24 GB VRAM).

---

## 1. Methodology & Evaluation Protocol

To rigorously evaluate the fine-tuned checkpoints, we establish a structured evaluation protocol.

### 1.1. Dataset & Split
*   **Dataset:** `unsloth/LaTeX_OCR` (images of handwritten/printed formulas mapped to LaTeX strings).
*   **Evaluation Set (N=30):** We increase the evaluation set to **30 representative samples** selected deterministically from the `test` split. This ensures a statistically more robust score than the initial 3-sample smoke check.
*   **Deterministic Decoding:** All evaluations are performed using greedy decoding (`do_sample=False`) to eliminate sampling noise.

### 1.2. Advanced LaTeX Normalization
LaTeX formatting is highly flexible (spaces, equivalent symbols, brace wrapping). To ensure fair scoring, we apply a **Normalization Pipeline** to both Ground Truth (GT) and Model Output before comparison:
1.  **Strip Spacing:** Remove all spaces (e.g., `a + b` -> `a+b`).
2.  **Delimiters Removal:** Strip standard delimiters (`$$`, `$`, `\(`).
3.  **Symbol Standardization:** Map equivalent or short-hand symbols:
    *   `\le` -> `\leq`
    *   `\ge` -> `\geq`
    *   `\to` -> `\rightarrow`
    *   `\epsilon` -> `\varepsilon`
4.  **Brace Standardization:** Standardize single-character subscripts and superscripts (e.g., `x_i` -> `x_{i}` or vice versa) using regular expressions.

### 1.3. Scoring Metrics
For each validation sample, we compute two scores:
1.  **Normalized Exact Match (EM):**
    *   `1.0` if the normalized model output matches the normalized GT exactly.
    *   `0.0` otherwise.
    *   *Significance:* Extremely strict; guarantees mathematical equivalence.
2.  **Normalized Edit Distance (NED):**
    *   Defined as: $\text{NED} = 1 - \frac{\text{LevenshteinDistance}(\text{clean\_pred}, \text{clean\_true})}{\max(\text{len}(\text{clean\_pred}), \text{len}(\text{clean\_true}))}$
    *   *Significance:* Provides a smooth metric between `0.0` and `1.0` for near-misses (e.g., a single typo in a long formula).

We report the **Average EM** and **Average NED** across the 30-sample evaluation set.

---

## 2. Proposed Experiment Matrix

We propose the following systematic sweep to isolate the impact of each hyperparameter. All runs are in **fp16** (`load_in_4bit=False`, `dtype=torch.float16`).

| Run ID | Sweep Focus | Vision Layers | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | Steps | Learning Rate | Expected Insights |
|---|---|---|---|---|---|---|---|
| **V0** | **Baseline** | — | — | — | — | — | Zero-shot capability of base `fp16` Gemma 4 E2B. |
| **V1_Trained** | **Reference** | Trained (`True`) | 16 | 16 | 60 | $2e-4$ | Standard high-LR vision SFT performance. |
| **V1_Frozen** | **Reference Frozen** | Frozen (`False`) | 16 | 16 | 60 | $2e-4$ | Isolates the impact of training vision encoder weights vs. language only. |
| **V2_Trained** | **High Capacity** | Trained (`True`) | 32 | 32 | 60 | $2e-4$ | Tests if higher LoRA rank prevents bottlenecking on complex symbols. |
| **V2_Frozen** | **High Capacity Frozen**| Frozen (`False`) | 32 | 32 | 60 | $2e-4$ | Symmetrical frozen control for the high capacity run. |
| **V3_Trained** | **Lower LR** | Trained (`True`) | 16 | 16 | 60 | $1e-4$ | Isolates the impact of a lower learning rate ($1e-4$ vs $2e-4$) under reference budget. |
| **V3_Frozen** | **Lower LR Frozen** | Frozen (`False`) | 16 | 16 | 60 | $1e-4$ | Symmetrical frozen control for the lower learning rate run. |
| **V4_Trained** | **Extended Budget** | Trained (`True`) | 16 | 16 | 120 | $2e-4$ | Studies over-fitting / training saturation on the OCR task. |
| **V4_Frozen** | **Extended Frozen** | Frozen (`False`) | 16 | 16 | 120 | $2e-4$ | Symmetrical frozen control for the extended budget run. |
| **V_Full_Trained**| **Full Epoch** | Trained (`True`) | 16 | 16 | None (1 Epoch) | $2e-4$ | Full 1-epoch training over the complete ~68k training set to establish peak capability. |
| **V_Full_Frozen** | **Full Epoch Frozen** | Frozen (`False`) | 16 | 16 | None (1 Epoch) | $2e-4$ | Symmetrical frozen control for the full epoch run. |

---

## 3. CLI Reference Commands

The experiments will be executable using CLI flags to ensure reproducibility.

```bash
# Run V0 (Baseline Eval only)
# (We will create examples/eval_vision.py for standalone evaluation)
uv run python examples/eval_vision.py --model unsloth/gemma-4-E2B-it --no-4bit

# Run V1_Trained (Reference SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 2e-4 --output-dir lora_vision_v1_trained

# Run V1_Frozen (Reference Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 2e-4 --finetune-vision-layers False --output-dir lora_vision_v1_frozen

# Run V2_Trained (High Capacity SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 32 --max-steps 60 --learning-rate 2e-4 --output-dir lora_vision_v2_trained

# Run V2_Frozen (High Capacity Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 32 --max-steps 60 --learning-rate 2e-4 --finetune-vision-layers False --output-dir lora_vision_v2_frozen

# Run V3_Trained (Lower LR SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 1e-4 --output-dir lora_vision_v3_trained

# Run V3_Frozen (Lower LR Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 1e-4 --finetune-vision-layers False --output-dir lora_vision_v3_frozen

# Run V4_Trained (Extended SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 120 --learning-rate 2e-4 --output-dir lora_vision_v4_trained

# Run V4_Frozen (Extended Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 120 --learning-rate 2e-4 --finetune-vision-layers False --output-dir lora_vision_v4_frozen

# Run V_Full_Trained (Full Epoch SFT - Vision Trained)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps -1 --epochs 1 --train-rows 0 --learning-rate 1e-4 --output-dir lora_vision_v_full_trained

# Run V_Full_Frozen (Full Epoch SFT - Vision Frozen)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps -1 --epochs 1 --train-rows 0 --learning-rate 1e-4 --finetune-vision-layers False --output-dir lora_vision_v_full_frozen
```

## 4. Experimental Results Table

We systematically record the performance metrics of each run here:

| Run ID | Sweep Focus | Vision Layers | Steps | Learning Rate | Avg EM (N=30) | Avg NED (N=30) | Peak VRAM | Training Time |
|---|---|---|---|---|---|---|---|---|
| **V0** | **Baseline** | — | — | — | **0.00%** | **71.57%** | ~10.5 GB | — |
| **V1_Trained** | **Reference** | Trained | 60 | $2e-4$ | **3.33%** | **72.22%** | ~11.7 GB | 2.5 min |
| **V1_Frozen** | **Reference Frozen**| Frozen | 60 | $2e-4$ | **0.00%** | **72.28%** | ~11.7 GB | 1.9 min |
| **V2_Trained** | **High Capacity** | Trained | 60 | $2e-4$ | **6.67%** | **71.34%** | ~11.7 GB | 2.5 min |
| **V2_Frozen** | **High Capacity Frz**| Frozen | 60 | $2e-4$ | **0.00%** | **73.39%** | ~11.7 GB | 1.9 min |
| **V3_Trained** | **Lower LR** | Trained | 60 | $1e-4$ | **0.00%** | **74.76%** | ~11.7 GB | 2.45 min |
| **V3_Frozen** | **Lower LR Frozen** | Frozen | 60 | $1e-4$ | **3.33%** | **75.03%** | ~11.7 GB | 1.9 min |
| **V4_Trained** | **Extended Budget** | Trained | 120 | $2e-4$ | **3.33%** | **65.63%** | ~11.7 GB | 4.7 min |
| **V4_Frozen** | **Extended Frozen** | Frozen | 120 | $2e-4$ | **3.33%** | **68.58%** | ~11.7 GB | 3.6 min |
| **V_Full_Trn**| **Full Epoch** | Trained | 1 Ep | $1e-4$ | | | | |
| **V_Full_Frz**| **Full Epoch Frozen**| Frozen | 1 Ep | $1e-4$ | | | | |

---

## 5. Symmetrical Sweeps with 560 Visual Token Budget

To study the impact of visual resolution on LaTeX OCR mathematical details, we rerun the sweeps at a doubled visual token budget of **560 tokens per image** (using `--vision-tokens 560`).

All runs are in full **fp16** (`load_in_4bit=False`, `dtype=torch.float16`) on the RTX 4090.

### 5.1. Symmetrical Sweeps Matrix (560 Tokens)

| Run ID | Sweep Focus | Vision Layers | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | Steps | Learning Rate | Expected Insights |
|---|---|---|---|---|---|---|---|
| **V0_560** | **Baseline** | — | — | — | — | — | Zero-shot VLM capability at double visual resolution. |
| **V1_Trained_560** | **Reference** | Trained (`True`) | 16 | 16 | 60 | $2e-4$ | Double visual token OCR SFT performance. |
| **V1_Frozen_560** | **Reference Frozen**| Frozen (`False`) | 16 | 16 | 60 | $2e-4$ | Symmetrical control with frozen vision. |
| **V2_Trained_560** | **High Capacity** | Trained (`True`) | 32 | 32 | 60 | $2e-4$ | Double tokens + higher LoRA capacity. |
| **V2_Frozen_560** | **High Capacity Frz**| Frozen (`False`) | 32 | 32 | 60 | $2e-4$ | Symmetrical control with frozen vision. |
| **V3_Trained_560** | **Lower LR** | Trained (`True`) | 16 | 16 | 60 | $1e-4$ | Slower step learning under double tokens. |
| **V3_Frozen_560** | **Lower LR Frozen** | Frozen (`False`) | 16 | 16 | 60 | $1e-4$ | Symmetrical control with frozen vision. |
| **V4_Trained_560** | **Extended Budget** | Trained (`True`) | 16 | 16 | 120 | $2e-4$ | Overfitting test under double visual tokens. |
| **V4_Frozen_560** | **Extended Frozen** | Frozen (`False`) | 16 | 16 | 120 | $2e-4$ | Symmetrical control with frozen vision. |

### 5.2. CLI Reference Commands (560 Tokens)

```bash
# Run V0_560 (Baseline Eval)
uv run python examples/eval_vision.py --model unsloth/gemma-4-E2B-it --no-4bit --eval-rows 30 --vision-tokens 560

# Run V1_Trained_560 (Reference SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 2e-4 --vision-tokens 560 --output-dir lora_vision_v1_trained_560

# Run V1_Frozen_560 (Reference Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 2e-4 --vision-tokens 560 --finetune-vision-layers False --output-dir lora_vision_v1_frozen_560

# Run V2_Trained_560 (High Capacity SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 32 --max-steps 60 --learning-rate 2e-4 --vision-tokens 560 --output-dir lora_vision_v2_trained_560

# Run V2_Frozen_560 (High Capacity Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 32 --max-steps 60 --learning-rate 2e-4 --vision-tokens 560 --finetune-vision-layers False --output-dir lora_vision_v2_frozen_560

# Run V3_Trained_560 (Lower LR SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 1e-4 --vision-tokens 560 --output-dir lora_vision_v3_trained_560

# Run V3_Frozen_560 (Lower LR Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 60 --learning-rate 1e-4 --vision-tokens 560 --finetune-vision-layers False --output-dir lora_vision_v3_frozen_560

# Run V4_Trained_560 (Extended SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 120 --learning-rate 2e-4 --vision-tokens 560 --output-dir lora_vision_v4_trained_560

# Run V4_Frozen_560 (Extended Frozen SFT)
uv run python examples/finetune_vision.py --no-4bit --lora-rank 16 --lora-alpha 16 --max-steps 120 --learning-rate 2e-4 --vision-tokens 560 --finetune-vision-layers False --output-dir lora_vision_v4_frozen_560
```

### 5.3. Experimental Results Table (560 Tokens)

| Run ID | Sweep Focus | Vision Layers | Steps | Learning Rate | Avg EM (N=30) | Avg NED (N=30) | Peak VRAM | Training Time |
|---|---|---|---|---|---|---|---|---|
| **V0_560** | **Baseline 560** | — | — | — | **3.33%** | **72.44%** | ~10.5 GB | — |
| **V1_Trained_560** | **Reference** | Trained | 60 | $2e-4$ | **0.00%** | **68.79%** | ~11.7 GB | 2.95 min |
| **V1_Frozen_560** | **Reference Frozen**| Frozen | 60 | $2e-4$ | **0.00%** | **74.43%** | ~11.7 GB | 2.0 min |
| **V2_Trained_560** | **High Capacity** | Trained | 60 | $2e-4$ | **6.67%** | **69.06%** | ~11.7 GB | 2.97 min |
| **V2_Frozen_560** | **High Capacity Frz**| Frozen | 60 | $2e-4$ | **3.33%** | **68.32%** | ~11.7 GB | 2.0 min |
| **V3_Trained_560** | **Lower LR** | Trained | 60 | $1e-4$ | **3.33%** | **74.36%** | ~11.7 GB | 3.1 min |
| **V3_Frozen_560** | **Lower LR Frozen** | Frozen | 60 | $1e-4$ | | | | |
| **V4_Trained_560** | **Extended Budget** | Trained | 120 | $2e-4$ | | | | |
| **V4_Frozen_560** | **Extended Frozen** | Frozen | 120 | $2e-4$ | | | | |


