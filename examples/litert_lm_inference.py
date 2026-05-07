"""Inference example using LiteRT-LM Python API.

Loads a LiteRT-LM model (.litertlm) and runs vision inference on a sample image,
rendering the resulting LaTeX in the terminal using TeXicode. Supports falling
back to text-only inference if the model does not support vision.

Usage:
    uv run python examples/litert_lm_inference.py --model path/to/model.litertlm
"""

import argparse
import os
import re
import litert_lm

DEFAULT_IMAGE = "examples/sample_latex.png"
INSTRUCTION = "Write the LaTeX representation for this image."


def extract_latex(text: str) -> str:
    # Try to extract from ```latex ... ```
    match_code = re.search(r"```latex\s*(.*?)\s*```", text, re.DOTALL)
    if match_code:
        return match_code.group(1).strip()

    # Try to extract from $$ ... $$
    match_display = re.search(r"\$\$\s*(.*?)\s*\$\$", text, re.DOTALL)
    if match_display:
        return match_display.group(1).strip()

    # Try to extract from $ ... $
    match_inline = re.search(r"\$\s*(.*?)\s*\$", text, re.DOTALL)
    if match_inline:
        return match_inline.group(1).strip()

    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to the LiteRT-LM (.litertlm) model file.",
    )
    parser.add_argument(
        "--image",
        "-i",
        default=DEFAULT_IMAGE,
        help=f"Path to image file (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=INSTRUCTION,
        help=f"User prompt (default: {INSTRUCTION!r}).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Initializing LiteRT-LM Engine with model: {args.model}...")
    has_vision = True
    
    # Robust initialization with FFI-safe fallback
    try:
        # Attempt to initialize with GPU vision backend
        engine = litert_lm.Engine(
            model_path=args.model,
            backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.GPU,
        )
        print("Initialized with GPU vision backend.")
    except RuntimeError:
        print("GPU vision initialization failed. Attempting CPU vision...")
        try:
            engine = litert_lm.Engine(
                model_path=args.model,
                backend=litert_lm.Backend.CPU,
                vision_backend=litert_lm.Backend.CPU,
            )
            print("Initialized with CPU vision backend.")
        except RuntimeError:
            print("Vision backend not supported or not found in model. Falling back to text-only mode.")
            has_vision = False
            engine = litert_lm.Engine(
                model_path=args.model,
                backend=litert_lm.Backend.CPU,
            )

    if has_vision and not os.path.exists(args.image):
        print(f"Warning: Image file {args.image} not found. Falling back to text-only.")
        has_vision = False

    with engine:
        print("Creating conversation...")
        with engine.create_conversation() as conversation:
            if has_vision:
                # Construct multimodal message
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image", "image": args.image},
                    ],
                }
                print(f"Sending multimodal prompt: {args.prompt}")
                print(f"Using image: {args.image}")
            else:
                # Text-only fallback prompt
                prompt = args.prompt if args.prompt != INSTRUCTION else "What is the capital of France?"
                user_message = {
                    "role": "user",
                    "content": prompt
                }
                print(f"Sending text prompt: {prompt}")

            # Send message and get response
            response = conversation.send_message(user_message)

            # Extract response text
            try:
                output_text = response["content"][0]["text"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected response format from LiteRT-LM: {response}. Error: {e}")

            print(f"\nModel Response:\n{output_text}")

            # Render LaTeX using TeXicode (only relevant if we generated LaTeX)
            try:
                import texicode.pipeline as tp

                cleaned_latex = extract_latex(output_text)
                if cleaned_latex != output_text or any(
                    c in cleaned_latex for c in ["\\", "^", "_", "{", "}"]
                ):
                    print(
                        f"\n\n{'=' * 40}\nRendered LaTeX (via TeXicode):\n{'=' * 40}"
                    )
                    rendered = tp.render_tex(
                        cleaned_latex, False, True, "raw", {"fonts": "normal"}
                    )
                    print(rendered)
                    print("=" * 40)
            except ImportError:
                print("\n(Install 'texicode' to render LaTeX in terminal)")
            except Exception as e:
                print(f"\n(Failed to render LaTeX via TeXicode: {e})")


if __name__ == "__main__":
    main()
