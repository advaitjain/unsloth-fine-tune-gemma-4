# Experiments

Documents the experiments run on `examples/finetune_gsm8k.py` to study how
training settings affect accuracy on a held-out GSM8K eval set. The
fine-tuning example itself is the subject; these experiments vary one or two
knobs at a time so you can see what each one buys you.

All experiments target the **6 GB VRAM** budget (Gemma 3 1B base in 4-bit,
QLoRA). Total wall-clock for the full sweep below is ~45 min on an RTX 4050
Laptop GPU.

## Eval prompts

Three GSM8K *test* problems (held out from the train split that fine-tuning
sees), hard-coded as `EVAL_PROMPTS` in `examples/finetune_gsm8k.py`:

1. **Janet's ducks** (expected: **$18**) — multi-step subtraction then
   multiplication.
2. **Robe bolts** (expected: **3**) — simple fraction arithmetic.
3. **Josh's house flip** (expected: **$70,000**) — multi-step accounting
   with a percentage gain that has to be netted against repair costs.

## Experiments and how to reproduce

Each command writes its LoRA adapter to a unique `--output-dir` so runs
don't clobber each other. The `BEFORE` and `AFTER` eval blocks are printed
to stdout as part of the training script.

### Reference run (the original example)

```bash
uv run python examples/finetune_gsm8k.py
```

Defaults: 60 steps, 1000 rows, `r=8`, `alpha=8`, `lr=2e-4`, linear schedule,
warmup=5. Output: `lora_gsm8k/`.

### Exp 1 — sampling fix only (no retraining)

Re-evaluate the reference adapter (and the base model) with greedy
decoding. Greedy is the default in `eval_gsm8k.py`; this experiment is just
running it.

```bash
# Base model alone
uv run python examples/eval_gsm8k.py

# The 60-step adapter from the reference run
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k/
```

### Exp 2 — more training, same hyperparameters

Train 5× longer on 2× more data; everything else at defaults.

```bash
uv run python examples/finetune_gsm8k.py \
  --max-steps 300 \
  --train-rows 2000 \
  --output-dir lora_gsm8k_exp2
```

```bash
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k_exp2/
```

### Exp 3 — higher LoRA rank + tuned hyperparameters

Quadruple the LoRA capacity, lower the learning rate, switch to cosine
schedule with longer warmup, train longer on more data.

```bash
uv run python examples/finetune_gsm8k.py \
  --max-steps 500 \
  --train-rows 3000 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --learning-rate 1e-4 \
  --warmup-steps 20 \
  --lr-scheduler-type cosine \
  --output-dir lora_gsm8k_exp3
```

```bash
uv run python examples/eval_gsm8k.py --adapter lora_gsm8k_exp3/
```

## Results

Accuracy on the three eval problems (✅ = correct final answer, ❌ = wrong).

| Run | P1: Janet's ducks ($18) | P2: Robe bolts (3) | P3: Josh's house ($70k) | Score | Final train loss | Wall clock |
|---|---|---|---|---|---|---|
| Reference (`temperature=1.0`, original eval) | $4096 ❌ | 11 ❌ | $57,500 ❌ | 0/3 | 0.92 | 3.3 min |
| **Base model, greedy** | (degenerate loop) ❌ | 3 ✅ (rambling) | $120,000 ❌ | ~1/3 | — | — |
| **Exp 1** — reference adapter, greedy | $26 ❌ | **3 ✅** | $120,000 ❌ | 1/3 | 0.92 | (no retrain) |
| **Exp 2** — 300 steps, r=8, greedy | **$18 ✅** | $4.5 ❌ | $10,000 ❌ | 1/3 | 0.62 | 14.2 min |
| **Exp 3** — 500 steps, r=32, lr=1e-4, cosine, greedy | **$18 ✅** | 5 ❌ | $120 ❌ | 1/3 | 0.57 | 22.9 min |

## Findings

1. **Greedy decoding is the single biggest accuracy lever.** The same 60-step
   adapter went from 0/3 → 1/3 just by switching `do_sample=False`. Sampling
   noise dominated the original run's failures.

2. **More training fixes Problem 1 but regresses Problem 2.** With 300+
   steps the model nails the harder arithmetic chain (`16-3-4=9`, `9×2=18`)
   but starts over-elaborating on the simple robe problem. Format-fitting
   and reasoning-fitting are partly at odds.

3. **Higher LoRA rank + tuned hyperparameters did not help.** Exp 3 had 4×
   the trainable parameters (26M vs 6.5M), lower lr, cosine schedule, more
   warmup, and 50% more training data — same 1/3 score, with a *worse* unit
   error on Problem 3. LoRA capacity is not the bottleneck at this scale.

4. **Problem 3 (the house flip) failed in every configuration.** Every
   variant computes the +$120k gain correctly but forgets to subtract the
   $50k repair cost from the profit. This is a capability ceiling of the
   1B base model, not a training-recipe problem.

For accuracy beyond ~1/3 on this eval, the next move is a larger base
model (e.g. `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`) or RL on a verifiable
reward — not more SFT compute on a 1B model.

## Training artifacts

After a training run, the directory passed to `--output-dir` contains:

| File / directory | What it is |
|---|---|
| `adapter_config.json` | LoRA configuration metadata: rank, alpha, target modules, base model id. PEFT reads this when loading the adapter. |
| `adapter_model.safetensors` | The trained LoRA weight deltas — the only file with the actual learned parameters. ~25 MB at `r=8`, ~100 MB at `r=32`. |
| `chat_template.jinja` | Saved chat template (Gemma's `<start_of_turn>user / model` format), in case you customize it. |
| `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` | Tokenizer files copied from the base model. Saved alongside the adapter so it can be loaded without separately re-fetching the base. |
| `README.md` | Auto-generated by `peft.save_pretrained` describing the adapter's hyperparameters. |
| `checkpoint-N/` | Intermediate trainer checkpoint at step `N` (mirrors the same files plus optimizer/scheduler state). HF Trainer keeps a small number by default. |

To use a saved adapter:

```bash
uv run python examples/eval_gsm8k.py --adapter <output-dir>
```

`FastModel.from_pretrained` reads `adapter_config.json`, pulls the matching
base model from the HF cache, and applies the adapter — no need to specify
the base model id separately.

## Available CLI flags

`examples/finetune_gsm8k.py`:

| Flag | Default | Purpose |
|---|---|---|
| `--model` / `-m` | `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` | Base model HF id. |
| `--max-steps` | 60 | Number of optimizer steps. |
| `--max-seq-length` | 2048 | Tokenizer / context window. |
| `--train-rows` | 1000 | GSM8K train subset size; 0 means full split (~7.5k). |
| `--output-dir` | `lora_gsm8k` | Where to save the LoRA adapter. |
| `--seed` | 3407 | Unsloth-canonical seed. |
| `--lora-rank` | 8 | LoRA rank. |
| `--lora-alpha` | 8 | LoRA alpha. Common: equal to rank, or 2× rank. |
| `--learning-rate` | 2e-4 | Peak learning rate. |
| `--warmup-steps` | 5 | LR warmup steps before scheduler kicks in. |
| `--lr-scheduler-type` | `linear` | HF scheduler name: `linear`, `cosine`, `constant`, etc. |

`examples/eval_gsm8k.py`:

| Flag | Default | Purpose |
|---|---|---|
| `--model` / `-m` | `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` | Base model HF id (used when `--adapter` is not set). |
| `--adapter` / `-a` | (none) | Path to a saved LoRA adapter dir; base is inferred from `adapter_config.json`. |
| `--max-new-tokens` | 512 | Generation cap per problem. |
| `--max-seq-length` | 2048 | Tokenizer / context window. |

## Gemma 4 E2B Systematic Experiments (N=50)

This section documents systematic evaluations of the Gemma 4 E2B model (4-bit and fp16 variants) on a 50-problem subset of the GSM8K test split using the automated evaluation tool (`examples/eval_gsm8k_automated.py`).

### Baseline Results

| Run | Base Model | Precision | GSM8K Test Accuracy (N=50) |
|---|---|---|---|
| B1 | `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` | 4-bit | 43/50 (86.00%) |
| B2 | `unsloth/gemma-4-E2B-it` | fp16 | 43/50 (86.00%) |

### SFT Tuning Sweeps

| Run | Base Model | Precision | Parameters | GSM8K Test Accuracy (N=50) | Final Loss | Wall Clock |
|---|---|---|---|---|---|---|

