import inspect
try:
    from unsloth.models.vision import get_peft_regex
    print("Path:", inspect.getsourcefile(get_peft_regex))
except Exception as e:
    print("Error:", e)
