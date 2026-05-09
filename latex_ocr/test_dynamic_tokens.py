import torch
from PIL import Image
from unsloth import FastVisionModel, get_chat_template

print("Loading model...")
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    load_in_4bit=True
)

print("Modifying token budget to 560...")
# 1. Modify Model Config
model.config.vision_soft_tokens_per_image = 560
model.config.vision_config.default_output_length = 560

# 2. Modify Processor Config
processor.image_processor.image_seq_length = 560
processor.image_processor.max_soft_tokens = 560
if hasattr(processor, "image_seq_length"):
    processor.image_seq_length = 560

print("Model Config vision_soft_tokens_per_image:", model.config.vision_soft_tokens_per_image)
print("Processor image_seq_length:", processor.image_seq_length)

# 3. Prepare inputs
processor = get_chat_template(processor, "gemma-4")
image = Image.new("RGB", (448, 448), color="blue")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

print("Inputs shapes:")
for k, v in inputs.items():
    print(f"  {k}: {v.shape}")

# 4. Forward Pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs)
print("Forward pass SUCCESS!")
print("Outputs logits shape:", outputs.logits.shape)
