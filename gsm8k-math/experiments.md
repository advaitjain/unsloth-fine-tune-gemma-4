# GSM8K Causal Math fine-tuning: Hyperparameter Sweeps

Audit sheet and metrics tracking fine-tuning experiments comparing supervised fine-tuning (SFT) configurations against the zero-shot baseline on the openai/gsm8k benchmark.

---

## Summary Metrics Board

Baseline and post-fine-tuning accuracy sweeps computed over **100 validation problems** from the held-out GSM8K main test split.

*   **Accuracy (EM)**: Percentage of target calculations that backtrace parser verified as correct (Exact Match on backtrack value).
*   **SFT Loss**: SFT optimization cross-entropy loss convergence values.

| Exp ID | Goals | Rank ($r$) | Alpha ($\alpha$) | LR | Scheduler | Train Rows | Steps | Baseline Acc | Post-SFT Acc | SFT Loss | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_0** | Base Model (Zero-shot control) | - | - | - | - | - | - | **84.00%** | — | — | Planned |
| **EXP_1** | Rank 32, Low split, Low LR | 32 | 64 | $5\times10^{-5}$ | Cosine | 1500 | 100 | **84.00%** | 78.00% | 0.9506 | Completed |
| **EXP_2** | Rank 32, Mid split, Low LR | 32 | 64 | $5\times10^{-5}$ | Cosine | 3000 | 200 | **84.00%** | 55.00% | 0.7057 | Completed |
| **EXP_3** | Rank 32, Max split, Low LR | 32 | 64 | $5\times10^{-5}$ | Cosine | 5000 | 300 | **84.00%** | 35.00% | 0.5910 | Completed |
| **EXP_4** | Rank 32, Low split, Mid LR | 32 | 64 | $1\times10^{-4}$ | Cosine | 1500 | 100 | **84.00%** | 64.00% | 0.8363 | Completed |
| **EXP_5** | Rank 32, Mid split, Mid LR | 32 | 64 | $1\times10^{-4}$ | Cosine | 3000 | 200 | **84.00%** | 34.00% | 0.6011 | Completed |
| **EXP_6** | Rank 32, Max split, Mid LR | 32 | 64 | $1\times10^{-4}$ | Cosine | 5000 | 300 | **84.00%** | 24.00% | 0.4895 | Completed |
| **EXP_7** | Rank 64, Low split, Low LR | 64 | 128| $5\times10^{-5}$ | Cosine | 1500 | 100 | **84.00%** | 72.00% | 0.8658 | Completed |
| **EXP_8** | Rank 64, Mid split, Low LR | 64 | 128| $5\times10^{-5}$ | Cosine | 3000 | 200 | **84.00%** | 42.00% | 0.6294 | Completed |
| **EXP_9** | Rank 64, Max split, Low LR | 64 | 128| $5\times10^{-5}$ | Cosine | 5000 | 300 | **84.00%** | 27.00% | 0.5212 | Completed |
| **EXP_10**| Rank 64, Low split, Mid LR | 64 | 128| $1\times10^{-4}$ | Cosine | 1500 | 100 | **84.00%** | 37.00% | 0.7567 | Completed |
| **EXP_11**| Rank 64, Mid split, Mid LR | 64 | 128| $1\times10^{-4}$ | Cosine | 3000 | 200 | **84.00%** | 36.00% | 0.5481 | Completed |
| **EXP_12**| Rank 64, Max split, Mid LR | 64 | 128| $1\times10^{-4}$ | Cosine | 5000 | 300 | **84.00%** | 25.00% | 0.4362 | Completed |
| **EXP_13**| Rank 64, Mid split, Low LR, Ext steps| 64 | 128| $5\times10^{-5}$ | Cosine | 3000 | 300 | **84.00%** | 31.00% | 0.5172 | Completed |
| **EXP_14**| Rank 64, Mid split, Mid LR, Ext steps| 64 | 128| $1\times10^{-4}$ | Cosine | 3000 | 300 | **84.00%** | 34.00% | 0.4380 | Completed |
| **EXP_15**| Rank 32, Max split, Safe LR control| 32 | 64 | $2\times10^{-5}$ | Cosine | 5000 | 300 | **84.00%** | 74.00% | 0.7489 | Completed |
| **EXP_16**| Rank 32, Low split, Safe LR check  | 32 | 64 | $1\times10^{-5}$ | Cosine | 1500 | 100 | **84.00%** | 81.00% | 1.4020 | Completed |
| **EXP_17**| Rank 32, Mid split, Safe LR check  | 32 | 64 | $1\times10^{-5}$ | Cosine | 3000 | 200 | **84.00%** | 83.00% | 1.0130 | Completed |
| **EXP_18**| Rank 32, Max split, Safe LR check  | 32 | 64 | $1\times10^{-5}$ | Cosine | 5000 | 300 | **84.00%** | 82.00% | 0.8748 | Completed |

---

## Detailed Sweeps Findings & Qualitative Observations

### [Sweep 1] Rank 32, Low split, Low LR (EXP_1)
*   **Observation**: Accuracy degraded from 84.00% to 78.00% (-6.00%).
*   **Analysis**: Fine-tuning on 1500 rows for 100 steps at $lr = 5\times10^{-5}$ reduced exact match accuracy compared to the zero-shot baseline. SFT loss converged to 0.9506.
*   **Sweep Direction**: Execute `EXP_2` (3000 rows, 200 steps) to measure the impact of increased training volume.

### [Sweep 2] Rank 32, Mid split, Low LR (EXP_2)
*   **Observation**: Accuracy degraded from 84.00% to 55.00% (-29.00%).
*   **Analysis**: Expanding training volume to 3000 rows for 200 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.7057. The fine-tuned model generated formatting that matched the training set but introduced factual errors in arithmetic (e.g., generating an $8 egg price in Problem 1, and altering percentage values in Problem 3).
*   **Sweep Direction**: Execute `EXP_3` (5000 rows, 300 steps) to measure the impact of maximum training volume.

### [Sweep 3] Rank 32, Max split, Low LR (EXP_3)
*   **Observation**: Accuracy degraded from 84.00% to 35.00% (-49.00%).
*   **Analysis**: Scaling data volume to 5000 rows for 300 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.5910. Factual arithmetic errors increased across the evaluation set (e.g., generating incorrect egg quantities in Problem 1 and modifying running rates in Problem 4).
*   **Sweep Direction**: Execute Group A's mid-LR sweeps starting with `EXP_4` ($lr = 1\times10^{-4}$) to evaluate accuracy at a higher learning rate.

### [Sweep 4] Rank 32, Low split, Mid LR (EXP_4)
*   **Observation**: Accuracy degraded from 84.00% to 64.00% (-20.00%).
*   **Analysis**: Fine-tuning on 1500 rows for 100 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.8363 and an exact match accuracy of 64.00%. This represents a larger accuracy reduction than the corresponding $5\times10^{-5}$ learning rate run (EXP_1, 78.00%).
*   **Qualitative Observation**: In Problem 3, the model integrated the repair cost directly into the final house value calculation.
*   **Sweep Direction**: Execute `EXP_5` (3000 rows, 200 steps) to observe accuracy scaling at this learning rate.

### [Sweep 5] Rank 32, Mid split, Mid LR (EXP_5)
*   **Observation**: Accuracy degraded from 84.00% to 34.00% (-50.00%).
*   **Analysis**: Fine-tuning on 3000 rows for 200 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.6011. Arithmetic errors included generating incorrect egg quantities in Problem 1 and altering sprint counts in Problem 4.
*   **Sweep Direction**: Execute `EXP_6` (5000 rows, 300 steps) to measure accuracy at maximum training volume for this learning rate.

### [Sweep 6] Rank 32, Max split, Mid LR (EXP_6)
*   **Observation**: Accuracy degraded from 84.00% to 24.00% (-60.00%).
*   **Analysis**: Fine-tuning on 5000 rows for 300 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.2882. Exact match accuracy on the test set was 24.00%.
*   **Sweep Direction**: Transition to Group B (Rank 64, Alpha 128) starting with `EXP_7` ($lr = 5\times10^{-5}$) to evaluate the impact of increased representation capacity.

### [Sweep 7] Rank 64, Low split, Low LR (EXP_7)
*   **Observation**: Accuracy degraded from 84.00% to 72.00% (-12.00%).
*   **Analysis**: Fine-tuning with $r=64, \alpha=128$ on 1500 rows for 100 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.8658. Accuracy was 72.00%, compared to 78.00% for the corresponding $r=32$ configuration (EXP_1).
*   **Qualitative Observation**: In Problem 4, the model generated the correct daily running rate (540), which was not observed in the corresponding EXP_1 run.
*   **Sweep Direction**: Execute `EXP_8` (3000 rows, 200 steps) to measure accuracy scaling at $r=64$.

### [Sweep 8] Rank 64, Mid split, Low LR (EXP_8)
*   **Observation**: Accuracy degraded from 84.00% to 42.00% (-42.00%).
*   **Analysis**: Fine-tuning with $r=64$ on 3000 rows for 200 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.6294. Exact match accuracy was 42.00%, compared to 55.00% for the corresponding $r=32$ configuration (EXP_2).
*   **Sweep Direction**: Execute `EXP_9` (5000 rows, 300 steps) to measure accuracy at maximum training volume for $r=64$.

### [Sweep 9] Rank 64, Max split, Low LR (EXP_9)
*   **Observation**: Accuracy degraded from 84.00% to 27.00% (-57.00%).
*   **Analysis**: Fine-tuning with $r=64$ on 5000 rows for 300 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.5212 and an exact match accuracy of 27.00%.
*   **Sweep Direction**: Transition to Group B's higher learning rate group starting with `EXP_10` ($lr = 1\times10^{-4}$) to evaluate accuracy under combined high capacity and learning rate.

### [Sweep 10] Rank 64, Low split, Mid LR (EXP_10)
*   **Observation**: Accuracy degraded from 84.00% to 37.00% (-47.00%).
*   **Analysis**: Fine-tuning with $r=64$ on 1500 rows for 100 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.7567 and an exact match accuracy of 37.00%, compared to 64.00% for the corresponding $r=32$ configuration (EXP_4).
*   **Qualitative Observation**: In Problem 3, the model calculated a 15% increase instead of 150%, resulting in an extracted value of 69500 against the target of 70000.
*   **Sweep Direction**: Execute `EXP_11` (3000 rows, 200 steps) to observe accuracy scaling under these parameters.

### [Sweep 11] Rank 64, Mid split, Mid LR (EXP_11)
*   **Observation**: Accuracy degraded from 84.00% to 36.00% (-48.00%).
*   **Analysis**: Fine-tuning with $r=64$ on 3000 rows for 200 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.5481 and an exact match accuracy of 36.00%.
*   **Qualitative Observation**: The model generated correct extracted values for 4 of the 5 demonstration problems (indices 1, 2, 3, and 5).
*   **Sweep Direction**: Execute `EXP_12` (5000 rows, 300 steps) to measure accuracy at maximum training volume for this configuration.

### [Sweep 12] Rank 64, Max split, Mid LR (EXP_12)
*   **Observation**: Accuracy degraded from 84.00% to 25.00% (-59.00%).
*   **Analysis**: Fine-tuning with $r=64$ on 5000 rows for 300 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.4362 and an exact match accuracy of 25.00%.
*   **Sweep Direction**: Transition to Group C (Extended steps & low-LR controls) starting with `EXP_13` ($r=64, lr = 5\times10^{-5}$, 3000 rows, 300 steps) to isolate the effect of extended training steps at a fixed volume.

### [Sweep 13] Rank 64, Mid split, Low LR, Ext steps (EXP_13)
*   **Observation**: Accuracy degraded from 84.00% to 31.00% (-53.00%).
*   **Analysis**: Fine-tuning on 3000 rows for 300 steps at $lr = 5\times10^{-5}$ resulted in a final SFT loss of 0.5172 and an exact match accuracy of 31.00%, compared to 42.00% for the 200-step run on the same data volume (EXP_8).
*   **Sweep Direction**: Execute `EXP_14` (3000 rows, 300 steps, $lr = 1\times10^{-4}$) to evaluate extended steps at a higher learning rate.

### [Sweep 14] Rank 64, Mid split, Mid LR, Ext steps (EXP_14)
*   **Observation**: Accuracy degraded from 84.00% to 34.00% (-50.00%).
*   **Analysis**: Fine-tuning on 3000 rows for 300 steps at $lr = 1\times10^{-4}$ resulted in a final SFT loss of 0.4380 and an exact match accuracy of 34.00%, compared to 36.00% for the 200-step run on the same data volume (EXP_11).
*   **Sweep Direction**: Execute `EXP_15` ($r=32, \alpha=64$, 5000 rows, 300 steps, $lr = 2\times10^{-5}$) to evaluate accuracy under a reduced learning rate control.

### [Sweep 15] Rank 32, Max split, Safe LR control (EXP_15)
*   **Observation**: Accuracy was 74.00% (-10.00% vs baseline).
*   **Analysis**: Fine-tuning with $r=32$ on 5000 rows for 300 steps at $lr = 2\times10^{-5}$ resulted in a final SFT loss of 0.7489 and an exact match accuracy of 74.00%, compared to 24.00% for the corresponding $lr = 1\times10^{-4}$ configuration (EXP_6).
*   **Qualitative Observation**: The model generated correct extracted values for 4 of the 5 demonstration problems (indices 1, 2, 3, and 4).
*   **Sweep Direction**: Execute low-LR sweeps ($1\times10^{-5}$) starting with `EXP_16` (1500 rows, 100 steps) to evaluate accuracy at a further reduced learning rate.

### [Sweep 16] Rank 32, Low split, Safe LR check (EXP_16)
*   **Observation**: Accuracy was 81.00% (-3.00% vs baseline).
*   **Analysis**: Fine-tuning with $r=32$ on 1500 rows for 100 steps at $lr = 1\times10^{-5}$ resulted in a final SFT loss of 1.4020 and an exact match accuracy of 81.00%.
*   **Qualitative Observation**: The model generated correct extracted values for all 5 demonstration problems.
*   **Sweep Direction**: Execute `EXP_17` (3000 rows, 200 steps) to observe accuracy scaling at this learning rate.

### [Sweep 17] Rank 32, Mid split, Safe LR check (EXP_17)
*   **Observation**: Accuracy was 83.00% (-1.00% vs baseline).
*   **Analysis**: Fine-tuning with $r=32$ on 3000 rows for 200 steps at $lr = 1\times10^{-5}$ resulted in a final SFT loss of 1.0130 and an exact match accuracy of 83.00%.
*   **Qualitative Observation**: The model generated correct extracted values for all 5 demonstration problems.
*   **Sweep Direction**: Execute `EXP_18` (5000 rows, 300 steps) to evaluate accuracy at maximum training volume for this learning rate.

### [Sweep 18] Rank 32, Max split, Safe LR check (EXP_18)
*   **Observation**: Accuracy was 82.00% (-2.00% vs baseline).
*   **Analysis**: Fine-tuning with $r=32$ on 5000 rows for 300 steps at $lr = 1\times10^{-5}$ resulted in a final SFT loss of 0.8748 and an exact match accuracy of 82.00%, compared to 24.00% for the corresponding $lr = 1\times10^{-4}$ configuration (EXP_6).
*   **Qualitative Observation**: The model integrated the fine-tuning formatting markers while generating correct extracted values across the evaluated sample set.
