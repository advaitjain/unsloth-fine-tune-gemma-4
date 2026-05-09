import os
import json
import subprocess
import sys
import time

RESULTS_FILE = "/usr/local/google/home/advaitjain/.gemini/jetski/brain/8e572895-10ea-46dd-a71a-2a5d1e1adc59/scratch/full_epoch_results.json"

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

results = {}

def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full-epoch checkpoint saved. Results: {list(results.keys())}")

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

def extract_metrics(output):
    em = None
    ned = None
    for line in output.split("\n"):
        if "Average Exact Match (EM):" in line:
            em = line.rsplit(":", 1)[-1].strip()
        if "Average Normalized Edit Dist (NED):" in line:
            ned = line.rsplit(":", 1)[-1].strip()
    return em, ned

epoch_runs = [
    {
        "id": "P2_280_Full_Epoch",
        "tokens": 280,
        "rank": 32,
        "alpha": 64,
        "lr": "5e-5",
        "output_dir": "lora_vision_p2_epoch_280"
    },
    {
        "id": "P2_560_Full_Epoch",
        "tokens": 560,
        "rank": 32,
        "alpha": 64,
        "lr": "5e-5",
        "output_dir": "lora_vision_p2_epoch_560"
    }
]

print("Starting Scaled Full-Epoch SFT Sweeps (280 vs 560 tokens)...")

for idx, run in enumerate(epoch_runs, 1):
    run_id = run["id"]
    print(f"\n================================================================================")
    print(f"[EPOCH {idx}/{len(epoch_runs)}] Executing Scaled Run: {run_id}")
    print(f"================================================================================")

    # 1. Training Arguments: 1 full epoch, max-steps=-1, train-rows=0 (all data)
    cmd_train = [
        "uv", "run", "python", "examples/finetune_vision.py",
        "--model", "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        "--no-4bit",
        "--lora-rank", str(run["rank"]),
        "--lora-alpha", str(run["alpha"]),
        "--learning-rate", run["lr"],
        "--vision-tokens", str(run["tokens"]),
        "--finetune-vision-layers", "True",
        "--finetune-language-layers", "True",
        "--max-steps", "-1",
        "--epochs", "1",
        "--train-rows", "0",
        "--output-dir", run["output_dir"]
    ]

    # 2. Evaluation Arguments: 50 test samples, matching token budget
    cmd_eval = [
        "uv", "run", "python", "examples/eval_vision.py",
        "--model", run["output_dir"],
        "--no-4bit",
        "--eval-rows", "50",
        "--vision-tokens", str(run["tokens"])
    ]

    try:
        # A. Execute Full-Epoch SFT Training
        print(f"Training {run_id} for 1 full epoch (~68k rows)...")
        _, train_time = run_cmd(cmd_train)

        # B. Execute SFT Evaluation
        print(f"Evaluating {run_id} SFT on 50 samples...")
        stdout, eval_time = run_cmd(cmd_eval)
        em, ned = extract_metrics(stdout)

        results[run_id] = {
            "type": "full_epoch_sft",
            "tokens": run["tokens"],
            "rank": run["rank"],
            "alpha": run["alpha"],
            "lr": run["lr"],
            "em": em,
            "ned": ned,
            "train_time_sec": train_time,
            "train_time_hr": round(train_time / 3600, 2),
            "eval_time_sec": eval_time
        }
        save_results()

        # C. Cleanup adapter checkpoint to conserve GPU disk space
        print(f"Cleaning up checkpoint weights for {run['output_dir']}...")
        subprocess.run(["rm", "-rf", run["output_dir"]])

    except Exception as e:
        print(f"CRITICAL ERROR running full-epoch SFT for {run_id}! Error: {e}")
        sys.exit(1)

print("\nScaled Full-Epoch SFT Grid Completed Successfully!")
