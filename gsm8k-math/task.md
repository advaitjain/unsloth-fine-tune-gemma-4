# Tasks: GSM8K-Math hyperparameter SFT Sweeps

Track development, syntax AST verification checks, smoke runs, background executions sweeps progress, auto-monitoring triggers, and demonstrator checks.

---

## Phase 1: Code Implementation
- [x] Write training pipeline engine `gsm8k-math/finetune_gsm8k.py`
- [x] Write standalone statistics evaluator `gsm8k-math/eval_gsm8k.py`
- [x] Write base-vs-sft demonstrator tool `gsm8k-math/inference_demo.py`
- [x] AST parse check validation placeholder

## Phase 2: Smoke Check & Verification
- [x] Run AST parse checks on scripts
- [x] Run mini-smoke training validation check (5 steps, 5 train, 5 eval)
- [x] Verify adapter saves logic inside `/tmp/`

## Phase 3: Multi-Config Sweeps (15 Experiments)
- [x] EXP_1 (Rank 32, LR 5e-5, 1500 rows, 100 steps)
- [x] EXP_2 (Rank 32, LR 5e-5, 3000 rows, 200 steps)
- [x] EXP_3 (Rank 32, LR 5e-5, 5000 rows, 300 steps)
- [x] EXP_4 (Rank 32, LR 1e-4, 1500 rows, 100 steps)
- [x] EXP_5 (Rank 32, LR 1e-4, 3000 rows, 200 steps)
- [x] EXP_6 (Rank 32, LR 1e-4, 5000 rows, 300 steps)
- [x] EXP_7 (Rank 64, LR 5e-5, 1500 rows, 100 steps)
- [x] EXP_8 (Rank 64, LR 5e-5, 3000 rows, 200 steps)
- [x] EXP_9 (Rank 64, LR 5e-5, 5000 rows, 300 steps)
- [x] EXP_10 (Rank 64, LR 1e-4, 1500 rows, 100 steps)
- [x] EXP_11 (Rank 64, LR 1e-4, 3000 rows, 200 steps)
- [x] EXP_12 (Rank 64, LR 1e-4, 5000 rows, 300 steps)
- [x] EXP_13 (Rank 64, LR 5e-5, 3000 rows, 300 steps, Extended steps)
- [x] EXP_14 (Rank 64, LR 1e-4, 3000 rows, 300 steps, Extended steps)
- [x] EXP_15 (Rank 32, LR 2e-5, 5000 rows, 300 steps, Stable LR control)
- [x] EXP_16 (Rank 32, LR 1e-5, 1500 rows, 100 steps, Low LR check)
- [x] EXP_17 (Rank 32, LR 1e-5, 3000 rows, 200 steps, Low LR check)
- [/] EXP_18 (Rank 32, LR 1e-5, 5000 rows, 300 steps, Low LR check)

## Phase 4: Validation & Demonstrator
- [x] Audit and locate 5 visual differences benchmarks
- [x] Lock visual indices targets inside `gsm8k-math/inference_demo.py`
- [x] Verify dynamic reloader and side-by-side demonstrator outputs
- [x] Final comparative reports updates
