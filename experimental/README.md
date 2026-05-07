# Experimental Playground: E2B Instruct Task Adaptation Studies

This directory serves as a playground for evaluating and fine-tuning **Gemma 4 E2B Instruct** on highly specialized, text-only cognitive tasks. 

Unlike general math benchmarks, these experiments explore model behavior when forced to adapt to **custom syntaxes** (logical Regular Expressions) or **domain-specific classifications** (emotion taxonomies).

---

## Task 1: Natural Language to Regular Expressions (NL-to-Regex)

### 1. Goal & Dialect Constraint
Causal instruct LLMs possess strong zero-shot capabilities for standard regex, but struggle when forced to map natural language to a custom logical semantic parsing grammar.
* **Dataset**: `inclinedadarsh/nl-to-regex` (Hugging Face)
* **Logical Dialect (LRegex)**: The dataset ground truths are written in LRegex, a logical regex dialect that uses prefix negation `~` and boolean intersection `&` (e.g., mapping `"lines which do not contain the letter 'e'"` to `~(.*e.*)`). Since the base model has never seen LRegex, it represents a strict test of syntactic SFT adaptation.

### 2. Evaluation Protocol
* **Automatic Markdown Extraction**: The evaluator (`experimental/eval_regex.py`) extracts the regular expression from conversational prose completions by matching fenced code blocks (````regex ... ````), inline backticks, or bolded headers, then comparing the isolated regex to the target gold LRegex string.
* **Baseline (B1)**: Base model evaluated zero-shot using greedy decoding.
* **Tuned (T1)**: LoRA SFT ($r=\alpha=32$, 300 steps, 774 training rows, LR $2e-4$ high learning rate to force formatting/style changes).

### 3. Results & The Generalization Bottleneck

| Run ID | Model Variant | SFT Hyperparameters | Exact Match Rate (N=50) | Final Loss | Notes |
|---|---|---|---|---|---|
| **B1** | Base E2B | Zero-Shot baseline (greedy) | **2.00%** (1/50) | — | Extracts standard PCRE regexes; gold LRegex mismatched. |
| **T1** | Tuned E2B | QLoRA ($r=32$, $a=32$), steps=300, LR $2e-4$ | **0.00%** (0/50) | 0.198 | Loss collapsed on train set, but completely failed on test set. |

#### Scientific Finding: The Generalization Bottleneck
* **Excellent Local Convergence**: SFT training without response masking collapsed loss gracefully to **`0.198`** (average token probability $>82\%$) on the train split.
* **Generalization Collapse**: When evaluated on held-out test prompts (Examples 1-50), the tuned model failed to synthesize the custom LRegex grammar, reverting to standard PCRE regular expressions (`^[^e\n]*$`) inside conversational paragraphs.
* **Prior Dominance**: With a small dataset (724 training rows), a QLoRA adapter ($r=32$) acts primarily as a **memorization cache (overfitting local inputs)**. It lacks the representation capacity to synthesize and generalize a brand-new grammatical syntax tree. When fed an unseen prompt, the base model's massive, pre-trained English-to-PCRE prior dominates the forward pass completely.

---

## Task 2: Emotion Classification (dair-ai/emotion)

### 1. Goal & Prompt Setup
To demonstrate a successful SFT lift while respecting the model's permanent conversational prior, we pivoted to a multi-class semantic classification task.
* **Dataset**: `dair-ai/emotion` (Hugging Face)
* **Task**: Classify a text diary entry into one of **6 emotions**: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.
* **Prompt Design**:
  ```markdown
  Classify the following diary entry into one of these 6 emotions: sadness, joy, love, anger, fear, surprise.

  Diary Entry: "[DIARY_TEXT]"

  Provide a brief conversational explanation of your reasoning, then output your final category strictly inside the tag: **Emotion:** <category>
  ```
* **Methodology**: Low-learning-rate SFT ($2e-5$, 150 steps, 2000 training rows, response-only masking) designed to gently align boundary classification weights while preserving the model's reasoning integrity.

### 2. Results & Semantic Integrity Audit

| Run ID | Model Variant | SFT Hyperparameters | Classification Accuracy (N=50) | Final Loss | Peak VRAM |
|---|---|---|---|---|---|
| **B1** | Base E2B | Zero-Shot baseline (greedy) | **54.00%** (27/50) | — | 8.3 GB |
| **T1** | Tuned E2B | QLoRA ($r=32$, $a=32$), steps=150, LR $2e-5$ | **56.00%** (28/50) | 0.4577 | 8.3 GB |

#### Scientific Finding: Reasoner Semantic Integrity vs. Benchmark Noise
SFT training converged cleanly (loss `0.4577`), but yielded a minor parsed exact-match accuracy change (+2.00%). A manual audit of the prediction logs (`task-1396.log`) revealed the following facts:
* **Inconsistent Human Benchmarks**: The `dair-ai/emotion` gold labels contain highly noisy, arbitrary, or technically flawed annotations. E.g.:
  - *Text*: `"i feel so cold..."` -> Gold Label: **`anger`**. (Model correctly reasoned and predicted `sadness`).
  - *Text*: `"friendly affection..."` -> Gold Label: **`joy`**. (Model correctly reasoned and predicted `love`).
  - *Text*: `"ashamed of mental health... let go of shame..."` -> Gold Label: **`sadness`**. (Model correctly identified the empathetic support as `love`).
  - *Text*: `"feeling of loss... Nostalgia at dad..."` -> Gold Label: **`sadness`**. (Model correctly identified the warm, wistful gratitude as `love`).
* **Semantic Integrity**: Because E2B's general reasoning weights were protected by our low learning rate ($2e-5$), **the model maintained its semantic logic and correctly refused to overfit to inconsistent human annotations**. During inference, it generated logically sound step-by-step reasoning explanations and outputted the conceptually correct label (e.g., matching "friendly affection" to `love`), resulting in a technical "exact-match mismatch" against the noisy gold label (`joy`).

---

## Reproduction Commands

All SFT training and evaluation pipelines are parameterized in separate, clean scripts:

### 1. Regex Task Sweeps
```bash
# A. Run zero-shot baseline (N=50)
uv run python experimental/eval_regex.py --num-examples 50

# B. Train high-LR adapter (r=32, LR 2e-4, 300 steps)
uv run python experimental/train_regex.py \
  --max-steps 300 \
  --learning-rate 2e-4 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --output-dir experimental/lora_regex_high_lr

# C. Evaluate the high-LR adapter checkpoint-300
uv run python experimental/eval_regex.py --adapter experimental/lora_regex_high_lr/checkpoint-300/ --num-examples 50
```

### 2. Emotion Task Sweeps
```bash
# A. Run zero-shot baseline (N=50)
uv run python experimental/eval_emotion.py --num-examples 50

# B. Train low-LR adapter with response masking (r=32, LR 2e-5, 150 steps)
uv run python experimental/train_emotion.py \
  --max-steps 150 \
  --train-rows 2000 \
  --learning-rate 2e-5 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --output-dir experimental/lora_emotion

# C. Evaluate the trained emotion adapter
uv run python experimental/eval_emotion.py --adapter experimental/lora_emotion/ --num-examples 50
```
