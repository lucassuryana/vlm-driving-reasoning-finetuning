"""
infer.py
Load a (fine-tuned or base) Qwen2-VL model and run one CARE-Drive inference example.
Works with base model out of the box — no fine-tuning required.

Usage:
    python src/infer.py --model Qwen/Qwen2-VL-7B-Instruct
    python src/infer.py --model ./outputs/qwen2vl-care-drive
"""

import argparse
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "You are an autonomous vehicle decision-making component designed to be "
    "responsive to human reasons. You will be given a traffic scene image and "
    "structured context data. Reason explicitly about human stakeholders and "
    "their expectations before making a decision."
)

EXAMPLE_CONTEXT = (
    "Context — Speed: 42 km/h | Distance to front vehicle: 8.5 m | "
    "Road marking: double_solid | Oncoming traffic: True | Weather: clear"
)

EXAMPLE_QUESTION = (
    "Given the traffic scene and context above, what should the AV do? "
    "Please reason through the relevant human stakeholders and their expectations "
    "before giving a final decision."
)

# Use a freely available test image URL; replace with a real scene path for actual runs
EXAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
# TODO: replace with actual driving scene image path, e.g. "data/raw/scene_001.jpg"


def run_inference(model_path: str, image_source: str, question: str) -> str:
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},
                {"type": "text", "text": f"{EXAMPLE_CONTEXT}\n\n{question}"},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the generated tokens (strip the input prompt)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen2-VL-7B-Instruct", help="Model path or HF hub ID"
    )
    parser.add_argument("--image", default=EXAMPLE_IMAGE_URL, help="Image path or URL")
    parser.add_argument("--question", default=EXAMPLE_QUESTION)
    args = parser.parse_args()

    response = run_inference(args.model, args.image, args.question)
    print("\n=== Model Response ===")
    print(response)
