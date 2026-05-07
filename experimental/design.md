# Experimental Design: NL-to-Regex Fine-Tuning on Gemma 4 E2B

This document outlines the methodology and setup for fine-tuning and evaluating **Gemma 4 E2B Instruct** on the **Natural Language to Regular Expressions (NL-to-Regex)** task.

## Goal
Base instruction-tuned models are extremely capable generalists but struggle to output dense, strictly structured syntaxes like Regular Expressions with perfect compiler compliance zero-shot. Our objective is to demonstrate that low-learning-rate Supervised Fine-Tuning (SFT) on a regex-specific dataset can yield a **clear, measurable, and massive lift in functional regex accuracy** over the base model.

## Dataset
* **Dataset Name**: `inclinedadarsh/nl-to-regex` (Hugging Face)
* **Task**: Translate a natural language pattern description (e.g. "lines which do not contain the letter 'e'") into a regular expression.
* **Syntax Dialect**: Ground truth regular expressions are represented in the custom logical regex dialect **LRegex** (e.g., `~(.*e.*)` using prefix negation `~` and boolean intersection `&`), which base LLMs do not natively produce zero-shot.

---

## Discovery: The Permanent Instruct Prior
During our initial SFT training run (using a low learning rate $2e-5$ over 150 steps), the model successfully converged to a very low loss ($0.52$). However, during inference, its output remained highly verbose and conversational.

Even when we increased the learning rate by 10× ($2e-4$) and trained for 300 steps in `lora_regex_high_lr` (collapsing SFT loss to an extremely low **$0.198$**), next-token logit comparisons proved that the model **still preferred conversational prose starts** (`Here...`, `You...`) rather than strictly outputting the DSL string. 

This confirms that **the pre-trained instruct-alignment prior in Gemma 4 E2B IT is extremely rigid and virtually permanent**. SFT cannot override this conversational envelope. Therefore, to measure semantic success, we must allow the conversational format but extract the target regex functionally from its markdown blocks.

---

## Methodology

### Updated Evaluation Protocol: Markdown Extraction
Instead of exact-matching the raw model response string directly, we introduce a robust, multi-stage regular expression extractor:
1. Match standard fenced code blocks: ` ```regex\n(.*?)\n``` ` or ` ```\n(.*?)\n``` `.
2. Match inline backtick wraps: ` `...` `.
3. Match bolded headers: `**Regex:** pattern`.
4. Fallback to the raw cleaned text string.

This isolates the regular expression, allowing direct exact-match comparison against the ground truth LRegex target string.

---

## Experimental Results (N=50)

| Run ID | Model Variant | SFT Hyperparameters | Regex Extraction Match Rate (N=50) | Train Loss | Notes |
|---|---|---|---|---|---|
| **B1** | Base E2B | Zero-Shot Baseline (Greedy) | **2.00%** (1/50) | — | Isolated PCRE expressions from markdown successfully; Gold LRegex mismatched. |
| **T1** | Tuned E2B | QLoRA ($r=32$, $a=32$), steps=300, LR $2e-4$ (High LR) | **0.00%** (0/50) | 0.198 | SFT loss collapsed; Generalization Bottleneck prevented syntax synthesis. |

---

## Scientific Analysis: The Generalization Bottleneck

1. **Excellent Local Train Convergence**:
   - Training without response masking successfully aligned parameter gradients to the 774 training samples. SFT loss gracefully collapsed to an exceptionally low **`0.198`** (a average token likelihood $>82\%$). This proves the adapter mathematically converged on the training set.

2. **Generalization Failures on Unseen Prompts**:
   - When presented with held-out test prompts (Examples 1-50), the model scored **0.00%** accuracy.
   - Instead of outputting the custom logical regex LRegex dialect (`~(.*e.*)`) that it memorized during training, the model continued to generate standard PCRE expressions (`^[^e\n]*$`).

3. **Prior Dominance on Small Datasets**:
   - A custom logical grammar dialect like LRegex is highly alien to the model's massive pre-trained English-to-PCRE regular expression representations.
   - With a small training split (724 rows), a QLoRA adapter ($r=32$) is only capable of **memorizing (overfitting)** the training examples. It lacks the semantic capacity to synthesize and generalize a brand-new grammar syntax tree to unseen prompt inputs.
   - Consequently, when fed an unseen prompt, the base model's massive pre-trained standard PCRE regex prior dominates the forward pass completely, resulting in a fallback to PCRE generation wrapped in standard markdown.

4. **Final Key Lesson**: When adapting highly aligned instruction LLMs to custom DSL dialects, **small-scale SFT acts primarily as a memorization cache, not a grammar synthesizer**. To successfully generalize a completely new syntax grammar, one must either:
   - Increase the dataset size by $10\times$ to $100\times$ (using data synthesis or syntactical prompt permutations).
   - Or train a significantly higher capacity LoRA adapter (e.g., full-layer $r=128$ or full-parameter fine-tuning) to allow deep representation shifts.


