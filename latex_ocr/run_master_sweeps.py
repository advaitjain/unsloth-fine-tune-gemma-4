import os
import json
import subprocess
import sys
import time

RESULTS_FILE = "/usr/local/google/home/advaitjain/.gemini/jetski/brain/8e572895-10ea-46dd-a71a-2a5d1e1adc59/scratch/master_results.json"

# Ensure target dir exists
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# Load existing results
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results from checkpoint.")
else:
    results = {}

# Helper to save results
def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Checkpoint saved. Total results: {len(results)}")

# Helper to run a command and log output
def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    start_time = time.time()
    res = subprocess.run(args, capture_output=True, text=True)
    runtime = time.time() - start_time
    if res.returncode != 0:
        print(f"ERROR executing command! Return code: {res.returncode}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return res.stdout, runtime

# Helper to extract metrics from eval output
def extract_metrics(output):
    # Parse evaluation summary:
    # Average Exact Match (EM):        3.33%
    # Average Normalized Edit Dist (NED): 72.44%
    em = None
    ned = None
    for line in output.split("\n"):
        if "Average Exact Match (EM):" in line:
            em = line.rsplit(":", 1)[-1].strip()
        if "Average Normalized Edit Dist (NED):" in line:
            ned = line.rsplit(":", 1)[-1].strip()
    return em, ned

# Define all experiments (Total 50 runs: 2 baselines + 48 SFT runs)
# Keys: Run ID, Tokens, SFT layers, Rank, Alpha, LR, TargetModules
sweeps = []

# --- 1. Baselines (Zero-shot) ---
sweeps.append({
    "id": "V0_280", "tokens": 280, "type": "baseline", "rank": None, "alpha": None, "lr": None,
    "layers": None, "ft_vision": False, "ft_lang": False, "target": ""
})
sweeps.append({
    "id": "V0_560", "tokens": 560, "type": "baseline", "rank": None, "alpha": None, "lr": None,
    "layers": None, "ft_vision": False, "ft_lang": False, "target": ""
})

# --- SFT Runs generator ---
tokens_list = [280, 560]
ranks = [16, 32]
lrs = ["2e-4", "1e-4", "5e-5"]

# SFT configurations
# Format: Config suffix, ft_vision, ft_lang, target_modules
configs = [
    ("Full", True, True, ""),
    ("VF", False, True, ""),
    ("LF", True, False, ""),
    ("PO", False, False, "embedding_projection")
]

for tk in tokens_list:
    for r in ranks:
        for lr in lrs:
            for cfg_suffix, ft_v, ft_l, tgt in configs:
                run_id = f"P2_{tk}_R{r}_LR{lr.replace('e-','').replace('.','')}_{cfg_suffix}"
                # Calculate expected Alpha natively
                alpha = r * 2
                sweeps.append({
                    "id": run_id, "tokens": tk, "type": "sft", "rank": r, "alpha": alpha, "lr": lr,
                    "layers": cfg_suffix, "ft_vision": ft_v, "ft_lang": ft_l, "target": tgt
                })

print(f"Configured {len(sweeps)} total master runs for the benchmark grid.")

# Execute the sweeps sequentially
for idx, run in enumerate(sweeps, 1):
    run_id = run["id"]
    print(f"\n================================================================================")
    print(f"[{idx}/{len(sweeps)}] Sweeping Run: {run_id}")
    print(f"================================================================================")

    if run_id in results:
        print(f"Run {run_id} already exists in checkpoint results. Skipping.")
        continue

    # --- STEP 1: Baseline / Zero-shot Evaluation ---
    if run["type"] == "baseline":
        print(f"Running Zero-Shot Baseline for {run['tokens']} tokens...")
        cmd = [
            "uv", "run", "python", "examples/eval_vision.py",
            "--model", "unsloth/gemma-4-E2B-it",
            "--no-4bit",
            "--eval-rows", "50",
            "--vision-tokens", str(run["tokens"])
        ]
        try:
            stdout, runtime = run_cmd(cmd)
            em, ned = extract_metrics(stdout)
            results[run_id] = {
                "type": "baseline",
                "tokens": run["tokens"],
                "em": em,
                "ned": ned,
                "train_time": 0,
                "eval_time": runtime
            }
            save_results()
        except Exception as e:
            print(f"CRITICAL ERROR evaluating baseline {run_id}! Error: {e}")
            sys.exit(1)

    # --- STEP 2: SFT Training & SFT Evaluation ---
    elif run["type"] == "sft":
        # SFT Training command
        output_dir = f"lora_vision_p2_{run_id.lower()}"
        cmd_train = [
            "uv", "run", "python", "examples/finetune_vision.py",
            "--model", "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
            "--no-4bit",
            "--lora-rank", str(run["rank"]),
            "--lora-alpha", str(run["alpha"]),
            "--learning-rate", run["lr"],
            "--vision-tokens", str(run["tokens"]),
            "--finetune-vision-layers", str(run["ft_vision"]),
            "--finetune-language-layers", str(run["ft_lang"]),
            "--max-steps", "60",
            "--output-dir", output_dir
        ]
        if run["target"]:
            cmd_train += ["--target-modules", run["target"]]

        # SFT Evaluation command
        cmd_eval = [
            "uv", "run", "python", "examples/eval_vision.py",
            "--model", output_dir,
            "--no-4bit",
            "--eval-rows", "50",
            "--vision-tokens", str(run["tokens"])
        ]

        try:
            # A. Train
            print(f"Running SFT Training for {run_id}...")
            _, train_time = run_cmd(cmd_train)
            
            # B. Evaluate
            print(f"Running Evaluation for {run_id}...")
            stdout, eval_time = run_cmd(cmd_eval)
            em, ned = extract_metrics(stdout)

            results[run_id] = {
                "type": "sft",
                "tokens": run["tokens"],
                "layers": run["layers"],
                "rank": run["rank"],
                "alpha": run["alpha"],
                "lr": run["lr"],
                "em": em,
                "ned": ned,
                "train_time": train_time,
                "eval_time": eval_time
            }
            save_results()
            
            # C. Cleanup checkpoint adapter weights locally to conserve disk space
            print(f"Cleaning up local checkpoint weights for {output_dir}...")
            subprocess.run(["rm", "-rf", output_dir])

        except Exception as e:
            print(f"CRITICAL ERROR running SFT sweep for {run_id}! Error: {e}")
            sys.exit(1)

print("\nBenchmark Symmetrical Grid Sweeps Execution Completed Successfully!")
