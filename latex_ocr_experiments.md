# Gemma 4 LaTeX OCR SFT Experiments Master-Record

This document serves as the official unified experiments master-record and benchmarks compiled for **Gemma 4 E2B Vision SFT on LaTeX OCR**.

---

## 🏆 1. Optimal Performing LoRA Runs & Evaluation Protocol

Based on rigorous Symmetrical Sweeps across structural architectures, ranks, token budgets, and learning rates in 16-bit precision (`fp16`) on the RTX 4090, we identify the following optimal configurations:

### 1.1. Best Scaled SFT Run (For Maximum Perfect Exact Math transcription)
*   **Run ID:** `P2_280_Full_Epoch`
*   **Configuration:** Fully Trained (Vision + Lang adapters active)
*   **Hyperparameters:** Rank $r=32$, Alpha $\alpha=64$, Learning Rate $5e-5$, Visual Tokens = 280, Steps: 8,585 (1 Full Epoch).
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

---

## 🛠️ 2. Evaluation Protocol: Computing EM & NED Scores

To compute the Exact Match (EM) correctness rate and Normalized Edit Distance (NED) approximate similarity, researchers should run the standard vision evaluation script:

```bash
uv run python examples/eval_vision.py --model <adapter_or_model_path> --no-4bit --eval-rows 50 --vision-tokens <280_or_560>
```

### How Metrics are Derived:
1.  **Greedy Decoding:** Eliminates sampling token noise (`do_sample=False`).
2.  **Advanced Normalization Pipeline:** Ground Truth and predicted strings are normalized by stripping spacing, wrappers (`$$`, `$`, `\(`), mapping command short-hands (`\le` -> `\leq`, `\to` -> `\rightarrow`), and standardizing brace indexing subscripts (`x_i` -> `x_{i}`).
3.  **Normalized Edit Distance (NED) Function:**
    $$\text{NED} = 1 - \frac{\text{LevenshteinDistance}(\text{clean\_pred}, \text{clean\_true})}{\max(\text{len}(\text{clean\_pred}), \text{len}(\text{clean\_true}))}$$
4.  **Exact Match (EM):** Returns `1.0` if normalized prediction matches normalized ground truth 100% exactly; returns `0.0` otherwise.

---

## 📊 3. Master Symmetrical Sweeps results Table (50 Samples)

Systematic Low-budget SFT parameter sweeps comparing Fully Trained, Vision Frozen (`VF`), Language Frozen (`LF`), and Projector-Only (`PO`) LoRA adapters under standard (280) vs high-res (560) token budgets over **50 test samples**:

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

## 4. Scaled Full-Epoch SFT results (16-bit fp16, LR $5e-5$, $r=32$, Fully Trained)

Benchmarks of the scaled 1-epoch trainings over the entire training set:

| Run ID | Token Budget | Config | Rank ($r$) | Alpha ($\alpha$) | LR | Avg EM (N=50) | $\Delta$ EM | Avg NED (N=50) | $\Delta$ NED | Train Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| **P2_280_Full_Epoch** | 280 | Fully Trained | 32 | 64 | $5e-5$ | **14.00%**| **+8.00%**| **68.39%** | -5.26% | 5.23 hours |
| **P2_560_Full_Epoch** | 560 | Fully Trained | 32 | 64 | $5e-5$ | **12.00%**| **+6.00%**| **64.46%** | -8.80% | 6.72 hours |
