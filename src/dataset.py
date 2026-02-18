dataset.py
Loads raw CARE-Drive annotations and converts them to Qwen2-VL
SFT message format. Each (image, question) → answer pair becomes
one independent training sample.
"""

import json
import base64
from pathlib import Path
from typing import Optional
from PIL import Image
import io

SYSTEM_PROMPT = (
    "You are an autonomous vehicle decision-making component designed to be "
    "responsive to human reasons. You will be given a traffic scene image and "
    "structured context data. Reason explicitly about human stakeholders and "
    "their expectations before making a decision."
)


def context_to_text(context: dict) -> str:
    """Convert structured context dict to a readable string prefix."""
    parts = []
    mapping = {
        "speed_kmh": "Speed: {} km/h",
        "distance_to_front_m": "Distance to front vehicle: {} m",
        "road_marking": "Road marking: {}",
        "oncoming_traffic": "Oncoming traffic: {}",
        "weather": "Weather: {}",
    }
    for key, template in mapping.items():
        if key in context:
            parts.append(template.format(context[key]))
    return "Context — " + " | ".join(parts) if parts else ""


def image_to_base64(image_path: str) -> Optional[str]:
    """Return base64-encoded JPEG string, or None if file missing."""
    p = Path(image_path)
    if not p.exists():
        return None
    with Image.open(p) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_image_content(image_path: str) -> dict:
    """
    Build the image block for a Qwen2-VL user message.
    Falls back to a path reference if image is not found (useful for dry runs).
    """
    b64 = image_to_base64(image_path)
    if b64:
        return {"type": "image", "image": f"data:image/jpeg;base64,{b64}"}
    # Fallback: local path reference (works with Qwen2-VL's processor if accessible)
    return {"type": "image", "image": str(Path(image_path).resolve())}


def annotation_to_samples(annotation: dict) -> list[dict]:
    """
    Convert one raw annotation (one image, N qa_pairs) into N training samples,
    each in Qwen2-VL chat messages format.
    """
    image_path = annotation["image_path"]
    context_text = context_to_text(annotation.get("context", {}))
    image_block = build_image_content(image_path)
    samples = []

    for qa in annotation["qa_pairs"]:
        user_text = f"{context_text}\n\n{qa['question']}".strip()
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        image_block,
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": qa["answer"]},
            ]
        }
        samples.append(sample)

    return samples


def load_dataset(json_path: str) -> list[dict]:
    """Load all annotations from a JSON file and return flat list of samples."""
    with open(json_path, "r") as f:
        annotations = json.load(f)
    samples = []
    for ann in annotations:
        samples.extend(annotation_to_samples(ann))
    return samples


def save_as_jsonl(samples: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    raw_path = "data/sample/sample_data.json"
    out_path = "data/processed/train.jsonl"
    samples = load_dataset(raw_path)
    print(f"Loaded {len(samples)} training samples from {raw_path}")
    for i, s in enumerate(samples[:2]):
        print(f"\n--- Sample {i} ---")
        print(f"  Q: {s['messages'][1]['content'][-1]['text'][:80]}...")
        print(f"  A: {s['messages'][2]['content'][:80]}...")
    save_as_jsonl(samples, out_path)
