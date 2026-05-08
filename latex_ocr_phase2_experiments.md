# Gemma 4 LaTeX OCR SFT: Master Phase 2 Symmetrical Sweeps (50-Run Grid)

This document outlines the master experimental design, methodology, and results for **Phase 2** of our vision SFT sweeps. We establish a highly rigorous, fully symmetrical grid comparing:
1.  **Fully Trained (Vision + Lang):** Both adapters active (`finetune_vision_layers=True`, `finetune_language_layers=True`).
2.  **Vision Frozen (Lang SFT Only):** `finetune_vision_layers=False`, `finetune_language_layers=True`.
3.  **Lang Frozen (Vision SFT Only):** `finetune_vision_layers=True`, `finetune_language_layers=False`.
4.  **Projector Only (PO SFT):** Both backbones frozen, targeting only the projector layer (`finetune_vision_layers=False`, `finetune_language_layers=False`, targeting `"embedding_projection"` explicitly).

Each configuration is swept across **both 280 and 560 visual token budgets** to capture the exact resolution performance delta.

---

## 1. Hyperparameter Standards

*   **Precision:** Full **16-bit (fp16)** (`--no-4bit`, `load_in_4bit=False`, `dtype=torch.float16`).
*   **Steps:** 60 steps (short-budget sweeps).
*   **Alpha Ratio:** Natively locked at **$\alpha = 2 \times \text{rank}$** ($\alpha=32$ for $r=16$; $\alpha=64$ for $r=32$).
*   **Evaluation Dataset:** **50 deterministic test samples** (greedy decoding) to establish high-precision, robust delta scores.
*   **Tuning Knobs (Unsloth PEFT):**
    *   `r` $\in \{16, 32\}$
    *   `lora_alpha` $\in \{32, 64\}$
    *   `lora_dropout = 0`
    *   `bias = "none"`
    *   `target_modules` $\in \{\text{None}, \text{["embedding_projection"]}\}$
    *   `random_state = 3407`

---

## 2. Complete Sweeps Grid (50 Runs)

We execute the following complete grid sequentially. All scores are compared directly to their respective zero-shot baseline (V0_280 or V0_560) to capture the exact performance delta ($\Delta$).

### 2.1. Part A: 280 Visual Token Sweeps (25 Runs)

| Run ID | SFT Configuration | Rank ($r$) | Alpha ($\alpha$) | LR | SFT Layers | Expected Insight |
|---|---|---|---|---|---|---|
| **V0_280** | **Baseline (Zero-shot)** | — | — | — | — | Establishes base V280 OCR capability. |
| **P2_280_R16_LR2_Full**| Fully Trained | 16 | 32 | $2e-4$ | Vision + Lang | V280 reference full SFT baseline. |
| **P2_280_R16_LR2_VF**  | Vision Frozen | 16 | 32 | $2e-4$ | Lang Only | Symmetrical V280 vision-frozen SFT. |
| **P2_280_R16_LR2_LF**  | Lang Frozen | 16 | 32 | $2e-4$ | Vision Only | Symmetrical V280 language-frozen SFT. |
| **P2_280_R16_LR2_PO**  | Projector Only| 16 | 32 | $2e-4$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_280_R16_LR1_Full**| Fully Trained | 16 | 32 | $1e-4$ | Vision + Lang | V280 reference stable LR full SFT. |
| **P2_280_R16_LR1_VF**  | Vision Frozen | 16 | 32 | $1e-4$ | Lang Only | Symmetrical V280 stable LR, vision frozen. |
| **P2_280_R16_LR1_LF**  | Lang Frozen | 16 | 32 | $1e-4$ | Vision Only | Symmetrical V280 stable LR, language frozen. |
| **P2_280_R16_LR1_PO**  | Projector Only| 16 | 32 | $1e-4$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_280_R16_LR5_Full**| Fully Trained | 16 | 32 | $5e-5$ | Vision + Lang | V280 reference ultra-stable LR full SFT. |
| **P2_280_R16_LR5_VF**  | Vision Frozen | 16 | 32 | $5e-5$ | Lang Only | Symmetrical V280 ultra-stable LR, vision frozen. |
| **P2_280_R16_LR5_LF**  | Lang Frozen | 16 | 32 | $5e-5$ | Vision Only | Symmetrical V280 ultra-stable LR, language frozen. |
| **P2_280_R16_LR5_PO**  | Projector Only| 16 | 32 | $5e-5$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_280_R32_LR2_Full**| Fully Trained | 32 | 64 | $2e-4$ | Vision + Lang | V280 high capacity, high LR full SFT. |
| **P2_280_R32_LR2_VF**  | Vision Frozen | 32 | 64 | $2e-4$ | Lang Only | Symmetrical V280 high capacity, vision frozen. |
| **P2_280_R32_LR2_LF**  | Lang Frozen | 32 | 64 | $2e-4$ | Vision Only | Symmetrical V280 high capacity, language frozen. |
| **P2_280_R32_LR2_PO**  | Projector Only| 32 | 64 | $2e-4$ | Projector Only| High capacity projector-only tuning. |
| **P2_280_R32_LR1_Full**| Fully Trained | 32 | 64 | $1e-4$ | Vision + Lang | V280 high capacity, stable LR full SFT. |
| **P2_280_R32_LR1_VF**  | Vision Frozen | 32 | 64 | $1e-4$ | Lang Only | Symmetrical V280 high capacity, stable LR, vision frozen. |
| **P2_280_R32_LR1_LF**  | Lang Frozen | 32 | 64 | $1e-4$ | Vision Only | Symmetrical V280 high capacity, stable LR, language frozen. |
| **P2_280_R32_LR1_PO**  | Projector Only| 32 | 64 | $1e-4$ | Projector Only| High capacity projector-only tuning. |
| **P2_280_R32_LR5_Full**| Fully Trained | 32 | 64 | $5e-5$ | Vision + Lang | V280 high capacity, ultra-stable LR full SFT. |
| **P2_280_R32_LR5_VF**  | Vision Frozen | 32 | 64 | $5e-5$ | Lang Only | Symmetrical V280 high capacity, vision frozen. |
| **P2_280_R32_LR5_LF**  | Lang Frozen | 32 | 64 | $5e-5$ | Vision Only | Symmetrical V280 high capacity, language frozen. |
| **P2_280_R32_LR5_PO**  | Projector Only| 32 | 64 | $5e-5$ | Projector Only| High capacity projector-only tuning. |

---

### 2.2. Part B: 560 Visual Token Sweeps (25 Runs)

| Run ID | SFT Configuration | Rank ($r$) | Alpha ($\alpha$) | LR | SFT Layers | Expected Insight |
|---|---|---|---|---|---|---|
| **V0_560** | **Baseline (Zero-shot)** | — | — | — | — | Establishes base V560 OCR capability. |
| **P2_560_R16_LR2_Full**| Fully Trained | 16 | 32 | $2e-4$ | Vision + Lang | V560 reference full SFT baseline. |
| **P2_560_R16_LR2_VF**  | Vision Frozen | 16 | 32 | $2e-4$ | Lang Only | Symmetrical V560 vision-frozen SFT. |
| **P2_560_R16_LR2_LF**  | Lang Frozen | 16 | 32 | $2e-4$ | Vision Only | Symmetrical V560 language-frozen SFT. |
| **P2_560_R16_LR2_PO**  | Projector Only| 16 | 32 | $2e-4$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_560_R16_LR1_Full**| Fully Trained | 16 | 32 | $1e-4$ | Vision + Lang | V560 reference stable LR full SFT. |
| **P2_560_R16_LR1_VF**  | Vision Frozen | 16 | 32 | $1e-4$ | Lang Only | Symmetrical V560 stable LR, vision frozen. |
| **P2_560_R16_LR1_LF**  | Lang Frozen | 16 | 32 | $1e-4$ | Vision Only | Symmetrical V560 stable LR, language frozen. |
| **P2_560_R16_LR1_PO**  | Projector Only| 16 | 32 | $1e-4$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_560_R16_LR5_Full**| Fully Trained | 16 | 32 | $5e-5$ | Vision + Lang | V560 reference ultra-stable LR full SFT. |
| **P2_560_R16_LR5_VF**  | Vision Frozen | 16 | 32 | $5e-5$ | Lang Only | Symmetrical V560 ultra-stable LR, vision frozen. |
| **P2_560_R16_LR5_LF**  | Lang Frozen | 16 | 32 | $5e-5$ | Vision Only | Symmetrical V560 ultra-stable LR, language frozen. |
| **P2_560_R16_LR5_PO**  | Projector Only| 16 | 32 | $5e-5$ | Projector Only| Fully locked towers, tuning projector. |
| **P2_560_R32_LR2_Full**| Fully Trained | 32 | 64 | $2e-4$ | Vision + Lang | V560 high capacity, high LR full SFT. |
| **P2_560_R32_LR2_VF**  | Vision Frozen | 32 | 64 | $2e-4$ | Lang Only | Symmetrical V560 high capacity, vision frozen. |
| **P2_560_R32_LR2_LF**  | Lang Frozen | 32 | 64 | $2e-4$ | Vision Only | Symmetrical V560 high capacity, language frozen. |
| **P2_560_R32_LR2_PO**  | Projector Only| 32 | 64 | $2e-4$ | Projector Only| High capacity projector-only tuning. |
| **P2_560_R32_LR1_Full**| Fully Trained | 32 | 64 | $1e-4$ | Vision + Lang | V560 high capacity, stable LR full SFT. |
| **P2_560_R32_LR1_VF**  | Vision Frozen | 32 | 64 | $1e-4$ | Lang Only | Symmetrical V560 high capacity, stable LR, vision frozen. |
| **P2_560_R32_LR1_LF**  | Lang Frozen | 32 | 64 | $1e-4$ | Vision Only | Symmetrical V560 high capacity, stable LR, language frozen. |
| **P2_560_R32_LR1_PO**  | Projector Only| 32 | 64 | $1e-4$ | Projector Only| High capacity projector-only tuning. |
| **P2_560_R32_LR5_Full**| Fully Trained | 32 | 64 | $5e-5$ | Vision + Lang | V560 high capacity, ultra-stable LR full SFT. |
| **P2_560_R32_LR5_VF**  | Vision Frozen | 32 | 64 | $5e-5$ | Lang Only | Symmetrical V560 high capacity, vision frozen. |
| **P2_560_R32_LR5_LF**  | Lang Frozen | 32 | 64 | $5e-5$ | Vision Only | Symmetrical V560 high capacity, language frozen. |
| **P2_560_R32_LR5_PO**  | Projector Only| 32 | 64 | $5e-5$ | Projector Only| High capacity projector-only tuning. |

---

## 3. Master Experimental Results Table (50 Samples)

We systematically compile all results and explicitly document the **relative delta ($\Delta$)** in transcription accuracy compared to its zero-shot starting baseline:

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

## 6. Scaled Full-Epoch SFT Results (16-bit fp16, LR $5e-5$, $r=32$, Fully Trained)

To benchmark the ultimate alignment capacity of our peak-performing configuration, we scaled the SFT budget to **1 full epoch** over all ~68k samples (effective batch size 8 = 8,585 steps) under both 280 and 560 visual token budgets:

| Run ID | Token Budget | Config | Rank ($r$) | Alpha ($\alpha$) | LR | Avg EM (N=50) | $\Delta$ EM | Avg NED (N=50) | $\Delta$ NED | Train Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| **V0_280** | 280 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.65%** | *Baseline* | — |
| **V0_560** | 560 | Baseline | — | — | — | **6.00%** | *Baseline* | **73.26%** | *Baseline* | — |
| **P2_280_Full_Epoch** | 280 | Fully Trained | 32 | 64 | $5e-5$ | **14.00%**| **+8.00%**| **68.39%** | -5.26% | 5.23 hours |
| **P2_560_Full_Epoch** | 560 | Fully Trained | 32 | 64 | $5e-5$ | **12.00%**| **+6.00%**| **64.46%** | -8.80% | 6.72 hours |

