#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "1. Installing LiteRT-LM compilation and runtime dependencies"
echo "============================================================"
uv pip install litert-lm-api
uv tool install litert-torch-nightly

echo -e "\n============================================================"
echo "2. Merging LoRA adapter (EXP_17) into base safetensors"
echo "============================================================"
uv run python litert-lm/merge_adapter.py \
  --adapter gsm8k-math/lora_exp17 \
  --output-dir litert-lm/merged_model

echo -e "\n============================================================"
echo "3. Exporting merged safetensors to LiteRT-LM format"
echo "============================================================"
mkdir -p litert-lm/compiled_model
litert-torch export_hf \
  --model=litert-lm/merged_model \
  --output_dir=litert-lm/compiled_model \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm

echo -e "\n============================================================"
echo "4. Running Qualitative 5-Sample Test via CLI"
echo "============================================================"
QUESTIONS=(
  "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
  "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
  "Josh decides to try flipping a house. He buys a house for \$80,000 and then puts in \$50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
  "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"
  "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
)

for idx in "${!QUESTIONS[@]}"; do
  echo -e "\n--- [Problem $((idx+1))] ---"
  echo "Q: ${QUESTIONS[$idx]}"
  echo "A:"
  litert-lm run litert-lm/compiled_model/model.litertlm --prompt="${QUESTIONS[$idx]}"
done

echo -e "\n============================================================"
echo "5. Running Quantitative 100-Sample Evaluation"
echo "============================================================"
uv run python litert-lm/eval_litert_gsm8k.py --model litert-lm/compiled_model/model.litertlm
