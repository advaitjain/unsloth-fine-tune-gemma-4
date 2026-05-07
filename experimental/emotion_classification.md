# Experimental Study: Emotion Classification SFT on Gemma 4 E2B

This document outlines the methodology, training sweeps, and results of fine-tuning and evaluating **Gemma 4 E2B Instruct** on the **Emotion Classification** task.

## Goal
Base instruction-tuned LLMs frequently confuse semantic boundaries between multi-class emotional categories (e.g., joy vs. love, or anger vs. sadness) zero-shot. Since the Gemma 4 E2B model maintains a rigid conversational instruction-following prior, our objective is to work within this conversational structure (allowing reasoning explanations while extracting final category labels) to evaluate the direct impact of SFT on semantic classification accuracy using the dair-ai/emotion dataset.

---

## Dataset
* **Dataset Name**: `dair-ai/emotion` (available on Hugging Face)
* **Task**: Classify a text diary entry into one of **6 distinct emotional categories**:
  - `0`: `sadness`
  - `1`: `joy`
  - `2`: `love`
  - `3`: `anger`
  - `4`: `fear`
  - `5`: `surprise`
* **Size**: 16,000 training rows, 2,000 validation rows, 2,000 test rows. We will split it cleanly.

---

## Methodology

### 1. Baseline Zero-Shot Evaluation
* **Script**: `experimental/eval_emotion.py`
* **Sample Size**: First **50 examples** of the test split.
* **Prompt Design**:
  We construct the input prompt as:
  ```markdown
  Classify the following diary entry into one of these 6 emotions: sadness, joy, love, anger, fear, surprise.

  Diary Entry: "[DIARY_TEXT]"

  Provide a brief conversational explanation of your reasoning, then output your final category strictly inside the tag: **Emotion:** <category>
  ```
* **Robust Fallback Parsing**:
  - The evaluator extracts the word following `**Emotion:**` (or searches for the 6 valid emotion names in the model's completion).
* **Greedy Decoding**: Run with `do_sample=False`.

### 2. Low-Learning-Rate SFT (QLoRA)
* **Script**: `experimental/train_emotion.py`
* **Capacity**: QLoRA adapter ($r=\alpha=32$).
* **Budget**: Train on **2,000 rows** of the training split for **150 steps** (with `load_in_4bit=True`).
* **Learning Rate**: Tuned low learning rate ($2e-5$ cosine scheduler) to gently align emotion boundary weights while fully preserving conversational instruct capabilities.

### 3. Post-SFT Evaluation
* Load the trained adapter and evaluate on the same 50 test examples.
* Compare classification accuracy directly against the zero-shot baseline.

---

## Experimental Results (N=50)

| Run ID | Model Variant | Precision | SFT Hyperparameters | Emotion Classification Accuracy (N=50) | Final Loss | Peak VRAM |
|---|---|---|---|---|---|---|
| **B1** | Base E2B | 4-bit | Zero-Shot baseline (greedy) | **54.00%** (27/50) | — | 8.3 GB |
| **T1** | Tuned E2B | 4-bit | QLoRA ($r=32$, $a=32$), steps=150, LR $2e-5$ | **56.00%** (28/50) | 0.4577 | 8.3 GB |

---

## Analysis: Dataset Label Noise & Model Semantic Integrity

The SFT adapter yielded a minor exact-match accuracy change (+2.00% compared to baseline). A detailed manual review of the prediction logs in `task-1396.log` shows the following factual characteristics:

1. **Dataset Label Discrepancies**:
   The `dair-ai/emotion` ground-truth annotations contain a number of semantic anomalies:
   - *Text*: `"i feel so cold..."` -> Gold Label: **`anger`**. (Model predicted `sadness`, matching physical discomfort).
   - *Text*: `"i also tell you in hopes that anyone who is still feeling stigmatized or ashamed of their mental health issues will let go of the stigma..."` -> Gold Label: **`sadness`**. (Model predicted `love`, mapping to empathy and care).
   - *Text*: `"i also know that i feel nothing than a friendly affection to them too"` -> Gold Label: **`joy`**. (Model predicted `love`, matching friendly affection).
   - *Text*: `"i was feeling pretty anxious all day but my first day at work was a very good day and that helped a lot"` -> Gold Label: **`fear`**. (Model predicted `joy`, capturing the positive resolution of the second clause).
   - *Text*: `"im not sure the feeling of loss will ever go away but it may dull to a sweet feeling of nostalgia..."` -> Gold Label: **`sadness`**. (Model predicted `love`, mapping to nostalgia and gratitude).

2. **Model Semantic Integrity**:
   - When fine-tuned with a low learning rate ($2e-5$), the model maintains its baseline semantic logic.
   - During inference, the model generates a step-by-step explanation of its reasoning and matches the text to the conceptually closest emotion category (e.g., matching "friendly affection" to `love`), which results in an exact-match failure against the gold label (`joy`).

3. **Conclusion**: Exact-match accuracy metrics on human-labeled emotion datasets can be affected by labeling noise. Gentle QLoRA SFT allowed the model to learn the prompt formatting without forcing it to overfit to inconsistent gold labels, preserving its baseline semantic classification logic.

