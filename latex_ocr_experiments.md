# Gemma 4 LaTeX OCR SFT Experiments Master-Record

This document serves as the official master-record of all low-budget sweeps, master symmetrical grid sweeps, and scaled full-epoch SFT benchmark investigations executed in 16-bit precision (`fp16`) for **Gemma 4 E2B Vision SFT on LaTeX OCR**.

---

## 🏆 1. Best Performing LoRA Runs Summary

Based on rigorous Symmetrical Sweeps across structural architectures, ranks, token budgets, and learning rates, we identified the following optimal configurations:

### 1.1. Best Scaled SFT Run (For Maximum Exact Mathematical transcription)
*   **Run ID:** `P2_280_Full_Epoch`
*   **Configuration:** Fully Trained (Vision + Lang adapters active)
*   **Hyperparameters:** Rank $r=32$, Alpha $\alpha=64$, Learning Rate $5e-5$, Visual Tokens = 280, Steps: 8,585 (1 Full Epoch over all ~68k samples).
*   **Avg EM (N=50):** **14.00%** (Zero-shot Baseline: 6.00% — **an absolute +8.00% EM improvement!**)
*   **Avg NED (N=50):** **68.39%**
*   **Command to Train:**
    ```bash
    uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 64 --learning-rate 5e-5 --vision-tokens 280 --max-steps -1 --epochs 1 --train-rows 0 --output-dir lora_vision_epoch_best
    ```

### 1.2. Best Short-Budget Sweep (For Maximum Balanced similarity)
*   **Run ID:** `P2_280_R32_LR55_Full`
*   **Configuration:** Fully Trained
*   **Hyperparameters:** Rank $r=32$, Alpha $\alpha=64$, Learning Rate $5e-5$, Visual Tokens = 280, Steps: 60.
*   **Avg EM (N=50):** **12.00%** (Baseline: 6.00% — **an absolute +6.00% EM improvement!**)
*   **Avg NED (N=50):** **78.14%** (Baseline: 73.65% — **an absolute +4.49% NED improvement!**)
*   **Command to Train:**
    ```bash
    uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 64 --learning-rate 5e-5 --vision-tokens 280 --max-steps 60 --output-dir lora_vision_sweep_best
    ```

### 1.3. Best Visual Resolution Sweep (Protected Feature Extraction)
*   **Run ID:** `P2_560_R32_LR55_VF`
*   **Configuration:** Vision Frozen (`VF` — Lang adapter active, Vision encoder 100% locked).
*   **Hyperparameters:** Rank $r=32$, Alpha $\alpha=64$, Learning Rate $5e-5$, Visual Tokens = 560, Steps: 60.
*   **Avg EM (N=50):** **6.00%**
*   **Avg NED (N=50):** **76.18%** (Baseline: 73.26% — **an absolute +2.92% NED improvement!**)
*   **Command to Train:**
    ```bash
    uv run python examples/finetune_vision.py --no-4bit --lora-rank 32 --lora-alpha 64 --learning-rate 5e-5 --vision-tokens 560 --finetune-vision-layers False --max-steps 60 --output-dir lora_vision_560_best
    ```

---

## 2. Methodology & Evaluation Protocol

To ensure high-precision metric compilation:
1.  **Deterministic Greedy Decoding:** Eliminates sampling noise (`do_sample=False`).
2.  **Advanced LaTeX Normalization:** Ground Truths (GT) and predictions are stripped of whitespace, display math wrapper wrappers (`$$`, `$`), equivalent commands unified (`\le` -> `\leq`, `\to` -> `\rightarrow`), and brace subscripts standardized (`x_i` -> `x_{i}`).
3.  **Scoring:** Normalized Exact Match (EM) for mathematical correctness, and Levenshtein-based Normalized Edit Distance (NED) for relative similarity.

---

## 3. Phase 1: Low Budget Symmetrical Sweeps (N=30 Validation Samples)

Phase 1 evaluated short-budget setups under the default visual sequence length of **280 visual tokens** on an evaluation set size of **30 samples**:

### 3.1. Symmetrical Phase 1 results Table

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

---

## 4. Phase 2: Master Symmetrical Benchmark Sweeps (N=50 Validation Samples)

Phase 2 expanded evaluation to **50 samples** and explored structural combinations across **280 vs 560 visual tokens**, locking LoRA Alpha scaling to `lora_alpha = 2 * lora_rank` natively, and adding Projector-Only (PO) training:

### 4.1. Master SFT Benchmark results Grid

| Run ID | Token Budget | SFT Config | Rank ($r$) | Alpha ($\alpha$) | LR | Avg EM (N=50) | $\Delta$ EM | Avg NED (N=50) | $\Delta$ NED |
|---|---|---|---|---|---|---|---|---|---|
| **V0_280** | 280 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.65%** | *Baseline* |
| **V0_560** | 560 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.26%** | *Baseline* |
| **P2_280_R16_LR2_Full** | 280 | Fully Trained | 16 | 32 | $2e-4$ | **8.00%** | +2.00% | **74.40%** | +0.75% |
| **P2_280_R16_LR2_VF**   | 280 | Vision Frozen | 16 | 32 | $2e-4$ | **8.00%** | +2.00% | **74.07%** | +0.42% |
| **P2_280_R16_LR2_LF**   | 280 | Lang Frozen   | 16 | 32 | $2e-4$ | **0.00%** | -6.00% | **12.68%** | -60.97% |
| **P2_280_R16_LR2_PO**   | 280 | Projector Only| 16 | 32 | $2e-4$ | **4.00%** | -2.00% | **72.66%** | -0.99% |
| **P2_280_R16_LR1_Full** | 280 | Fully Trained | 16 | 32 | $1e-4$ | **10.00%**| +4.00% | **76.61%** | +2.96% |
| **P2_280_R16_LR1_VF**   | 280 | Vision Frozen | 16 | 32 | $1e-4$ | **10.00%**| +4.00% | **75.25%** | +1.60% |
| **P2_280_R16_LR1_LF**   | 280 | Lang Frozen   | 16 | 32 | $1e-4$ | **4.00%** | -2.00% | **72.20%** | -1.45% |
| **P2_280_R16_LR1_PO**   | 280 | Projector Only| 16 | 32 | $1e-4$ | **6.00%** | 0.00% | **73.31%** | -0.34% |
| **P2_280_R16_LR5_Full** | 280 | Fully Trained | 16 | 32 | $5e-5$ | **8.00%** | +2.00% | **74.78%** | +1.13% |
| **P2_280_R16_LR5_VF**   | 280 | Vision Frozen | 16 | 32 | $5e-5$ | **8.00%** | +2.00% | **74.24%** | +0.59% |
| **P2_280_R16_LR5_LF**   | 280 | Lang Frozen   | 16 | 32 | $5e-5$ | **8.00%** | +2.00% | **74.41%** | +0.76% |
| **P2_280_R16_LR5_PO**   | 280 | Projector Only| 16 | 32 | $5e-5$ | **6.00%** | 0.00% | **73.98%** | +0.33% |
| **P2_280_R32_LR2_Full** | 280 | Fully Trained | 32 | 64 | $2e-4$ | **8.00%** | +2.00% | **71.69%** | -1.96% |
| **P2_280_R32_LR2_VF**   | 280 | Vision Frozen | 32 | 64 | $2e-4$ | **8.00%** | +2.00% | **73.27%** | -0.38% |
| **P2_280_R32_LR2_LF**   | 280 | Lang Frozen   | 32 | 64 | $2e-4$ | **0.00%** | -6.00% | **10.22%** | -63.43% |
| **P2_280_R32_LR2_PO**   | 280 | Projector Only| 32 | 64 | $2e-4$ | **8.00%** | +2.00% | **71.50%** | -2.15% |
| **P2_280_R32_LR1_Full** | 280 | Fully Trained | 32 | 64 | $1e-4$ | **8.00%** | +2.00% | **73.62%** | -0.03% |
| **P2_280_R32_LR1_VF**   | 280 | Vision Frozen | 32 | 64 | $1e-4$ | **8.00%** | +2.00% | **74.10%** | +0.45% |
| **P2_280_R32_LR1_LF**   | 280 | Lang Frozen   | 32 | 64 | $1e-4$ | **0.00%** | -6.00% | **46.78%** | -26.87% |
| **P2_280_R32_LR1_PO**   | 280 | Projector Only| 32 | 64 | $1e-4$ | **6.00%** | 0.00% | **73.70%** | +0.05% |
| **P2_280_R32_LR5_Full** | 280 | Fully Trained | 32 | 64 | $5e-5$ | **12.00%**| +6.00% | **78.14%** | +4.49% |
| **P2_280_R32_LR5_VF**   | 280 | Vision Frozen | 32 | 64 | $5e-5$ | **8.00%** | +2.00% | **75.58%** | +1.93% |
| **P2_280_R32_LR5_LF**   | 280 | Lang Frozen   | 32 | 64 | $5e-5$ | **10.00%**| +4.00% | **74.26%** | +0.61% |
| **P2_280_R32_LR5_PO**   | 280 | Projector Only| 32 | 64 | $5e-5$ | **6.00%** | 0.00% | **73.77%** | +0.12% |
| **P2_560_R16_LR2_Full** | 560 | Fully Trained | 16 | 32 | $2e-4$ | **8.00%** | +2.00% | **73.35%** | +0.09% |
| **P2_560_R16_LR2_VF**   | 560 | Vision Frozen | 16 | 32 | $2e-4$ | **6.00%** | 0.00% | **68.82%** | -4.44% |
| **P2_560_R16_LR2_LF**   | 560 | Lang Frozen   | 16 | 32 | $2e-4$ | **0.00%** | -6.00% | **47.19%** | -26.07% |
| **P2_560_R16_LR2_PO**   | 560 | Projector Only| 16 | 32 | $2e-4$ | **6.00%** | 0.00% | **73.34%** | +0.08% |
| **P2_560_R16_LR1_Full** | 560 | Fully Trained | 16 | 32 | $1e-4$ | **6.00%** | 0.00% | **74.44%** | +1.18% |
| **P2_560_R16_LR1_VF**   | 560 | Vision Frozen | 16 | 32 | $1e-4$ | **6.00%** | 0.00% | **75.47%** | +2.21% |
| **P2_560_R16_LR1_LF**   | 560 | Lang Frozen   | 16 | 32 | $1e-4$ | **2.00%** | -4.00% | **67.97%** | -5.29% |
| **P2_560_R16_LR1_PO**   | 560 | Projector Only| 16 | 32 | $1e-4$ | **6.00%** | 0.00% | **73.67%** | +0.41% |
| **P2_560_R16_LR5_Full** | 560 | Fully Trained | 16 | 32 | $5e-5$ | **6.00%** | 0.00% | **75.49%** | +2.23% |
| **P2_560_R16_LR5_VF**   | 560 | Vision Frozen | 16 | 32 | $5e-5$ | **6.00%** | 0.00% | **75.69%** | +2.43% |
| **P2_560_R16_LR5_LF**   | 560 | Lang Frozen   | 16 | 32 | $5e-5$ | **4.00%** | -2.00% | **74.04%** | +0.78% |
| **P2_560_R16_LR5_PO**   | 560 | Projector Only| 16 | 32 | $5e-5$ | **6.00%** | 0.00% | **73.36%** | +0.10% |
| **P2_560_R32_LR2_Full** | 560 | Fully Trained | 32 | 64 | $2e-4$ | **6.00%** | 0.00% | **71.12%** | -2.14% |
| **P2_560_R32_LR2_VF**   | 560 | Vision Frozen | 32 | 64 | $2e-4$ | **6.00%** | 0.00% | **66.38%** | -6.88% |
| **P2_560_R32_LR2_LF**   | 560 | Lang Frozen   | 32 | 64 | $2e-4$ | **0.00%** | -6.00% | **17.38%** | -55.88% |
| **P2_560_R32_LR2_PO**   | 560 | Projector Only| 32 | 64 | $2e-4$ | **6.00%** | 0.00% | **72.59%** | -0.67% |
| **P2_560_R32_LR1_Full** | 560 | Fully Trained | 32 | 64 | $1e-4$ | **8.00%** | +2.00% | **72.64%** | -0.62% |
| **P2_560_R32_LR1_VF**   | 560 | Vision Frozen | 32 | 64 | $1e-4$ | **6.00%** | 0.00% | **71.29%** | -1.97% |
| **P2_560_R32_LR1_LF**   | 560 | Lang Frozen   | 32 | 64 | $1e-4$ | **2.00%** | -4.00% | **59.55%** | -13.71% |
| **P2_560_R32_LR1_PO**   | 560 | Projector Only| 32 | 64 | $1e-4$ | **6.00%** | 0.00% | **72.71%** | -0.55% |
| **P2_560_R32_LR5_Full** | 560 | Fully Trained | 32 | 64 | $5e-5$ | **4.00%** | -2.00% | **75.35%** | +2.09% |
| **P2_560_R32_LR5_VF**   | 560 | Vision Frozen | 32 | 64 | $5e-5$ | **6.00%** | 0.00% | **76.18%** | +2.92% |
| **P2_560_R32_LR5_LF**   | 560 | Lang Frozen   | 32 | 64 | $5e-5$ | **4.00%** | -2.00% | **71.80%** | -1.46% |
| **P2_560_R32_LR5_PO**   | 560 | Projector Only| 32 | 64 | $5e-5$ | **6.00%** | 0.00% | **73.13%** | -0.13% |

---

## 5. Scaled Full-Epoch SFT Results (16-bit fp16, LR $5e-5$, $r=32$, Fully Trained)

To benchmark the ultimate alignment capacity of our peak-performing configuration, we scaled the SFT SFT budget to **1 full epoch** over all ~68k samples (effective batch size 8 = 8,585 steps) under both 280 and 560 visual token budgets:

| Run ID | Token Budget | Config | Rank ($r$) | Alpha ($\alpha$) | LR | Avg EM (N=50) | $\Delta$ EM | Avg NED (N=50) | $\Delta$ NED | Train Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| **V0_280** | 280 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.65%** | *Baseline* | — |
| **V0_560** | 560 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.26%** | *Baseline* | — |
| **P2_280_Full_Epoch** | 280 | Fully Trained | 32 | 64 | $5e-5$ | **14.00%**| **+8.00%**| **68.39%** | -5.26% | 5.23 hours |
| **P2_560_Full_Epoch** | 560 | Fully Trained | 32 | 64 | $5e-5$ | **12.00%**| **+6.00%**| **64.46%** | -8.80% | 6.72 hours |
