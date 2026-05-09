from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    load_in_4bit=True
)

print("Processor type:", type(processor))
print("Processor attributes:", dir(processor))
if hasattr(processor, "image_processor"):
    print("Image Processor type:", type(processor.image_processor))
    print("Image Processor attributes:", dir(processor.image_processor))
    print("Image Processor config:", processor.image_processor.__dict__)
