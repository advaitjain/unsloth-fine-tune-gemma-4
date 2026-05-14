#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="litert-lm"
RUN_BATCH_EVAL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-eval)
      RUN_BATCH_EVAL=true
      shift
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

START_TOTAL=$SECONDS

echo "============================================================"
echo "1. Installing LiteRT-LM compilation and runtime dependencies"
echo "============================================================"
STEP_START=$SECONDS
uv pip install litert-lm-api
uv tool install litert-torch-nightly
uv tool install ai-edge-quantizer-nightly
echo ">>> Step 1 completed in $((SECONDS - STEP_START)) seconds. <<<"

echo -e "\n============================================================"
echo "2. Merging LoRA adapter (EXP_17) into base safetensors"
echo "============================================================"
STEP_START=$SECONDS
uv run python litert-lm/merge_adapter.py \
  --adapter gsm8k-math/lora_exp17 \
  --output-dir "$OUTPUT_DIR/merged_model"
echo ">>> Step 2 completed in $((SECONDS - STEP_START)) seconds. <<<"

echo -e "\n============================================================"
echo "3a. Exporting merged safetensors to LiteRT-LM format. Use empty quantization recipe to get a float model."
echo "============================================================"
STEP_START=$SECONDS
mkdir -p "$OUTPUT_DIR/compiled_model"
litert-torch export_hf \
  --model="$OUTPUT_DIR/merged_model" \
  --output_dir="$OUTPUT_DIR/compiled_model" \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm \
  --quantization_recipe=""
echo ">>> Step 3a completed in $((SECONDS - STEP_START)) seconds. <<<"

echo -e "\n============================================================"
echo "3b. Applying AI Edge Quantizer (aeq) schemes to compiled model"
echo "============================================================"
STEP_START=$SECONDS
RECIPES=("dynamic_wi8c_afp32" "gemma4_mixed48_hr" "gemma4_mixed48_b32" "gemma4_mixed48_b64")
for recipe in "${RECIPES[@]}"; do
  RECIPE_START=$SECONDS
  echo ">>> Quantizing model with recipe: $recipe <<<"
  uv run aeq --model_file "$OUTPUT_DIR/compiled_model/model.litertlm" --recipe="$recipe" --output_dir "$OUTPUT_DIR/compiled_model" --overwrite_outputs
  echo ">>> Recipe $recipe completed in $((SECONDS - RECIPE_START)) seconds. <<<"
done
echo ">>> Step 3b total completed in $((SECONDS - STEP_START)) seconds. <<<"

echo -e "\n============================================================"
echo "4. Running Qualitative 5-Sample Test via CLI across all models"
echo "============================================================"
STEP_START=$SECONDS
QUESTIONS=(
  "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
  "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
  "Josh decides to try flipping a house. He buys a house for \$80,000 and then puts in \$50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
  "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"
  "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
)

for model_file in "$OUTPUT_DIR"/compiled_model/*.litertlm; do
  MODEL_START=$SECONDS
  echo -e "\n>>> Evaluating model: $(basename "$model_file") <<<"
  for idx in "${!QUESTIONS[@]}"; do
    echo -e "\n--- [Problem $((idx+1))] ---"
    echo "Q: ${QUESTIONS[$idx]}"
    echo "A:"
    litert-lm run "$model_file" --prompt="${QUESTIONS[$idx]}"
  done
  echo ">>> Model evaluation completed in $((SECONDS - MODEL_START)) seconds. <<<"
done
echo ">>> Step 4 completed in $((SECONDS - STEP_START)) seconds. <<<"

if [ "$RUN_BATCH_EVAL" = true ]; then
  echo -e "\n============================================================"
  echo "5. Running Quantitative 100-Sample Evaluation across all models"
  echo "============================================================"
  STEP_START=$SECONDS
  for model_file in "$OUTPUT_DIR"/compiled_model/*.litertlm; do
    EVAL_START=$SECONDS
    echo -e "\n>>> Running batch evaluation for: $(basename "$model_file") <<<"
    uv run python litert-lm/eval_litert_gsm8k.py --model "$model_file"
    echo ">>> Batch evaluation completed in $((SECONDS - EVAL_START)) seconds. <<<"
  done
  echo ">>> Step 5 completed in $((SECONDS - STEP_START)) seconds. <<<"
fi

echo -e "\n============================================================"
echo ">>> Total pipeline execution completed in $((SECONDS - START_TOTAL)) seconds. <<<"
echo "============================================================"
