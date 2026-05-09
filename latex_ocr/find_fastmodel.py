with open(".venv/lib/python3.11/site-packages/unsloth/models/vision.py", "r") as f:
    for idx, line in enumerate(f, 1):
        if "def get_peft_regex" in line:
            print(f"Line {idx}: {line.strip()}")
