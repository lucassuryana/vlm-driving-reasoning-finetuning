# reason-driven-av

**Fine-tuning Qwen2-VL for Reason-Driven Autonomous Vehicle Decision Making**

> 🚧 **Work in Progress** — Active development. Core pipeline functional; training and evaluation ongoing.

---

Expected repo tree:
```text
reason-driven-av/
├── README.md
├── requirements.txt
├── configs/
│   └── training.yaml
├── data/
│   ├── raw/                  # original images + annotation JSONs
│   ├── processed/            # converted message-format JSONL
│   └── sample/
│       └── sample_data.json  # tiny example for smoke-testing
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── infer.py
├── notebooks/
│   └── explore_dataset.ipynb
├── scripts/
│   └── prepare_data.sh
└── assets/
    └── pipeline_overview.png  # placeholder for diagram
```

## Overview

This repository implements a supervised fine-tuning (SFT) pipeline for **Qwen2-VL**, a vision-language model (VLM), to produce autonomous vehicle (AV) decisions that are explicitly grounded in **human-relevant reasons** — rather than pattern-matched from visual input alone.

The project extends **CARE-Drive** *(Context-Aware Reasons Evaluation for Driving)*, a framework for evaluating whether VLMs respond appropriately to structured normative contextual factors (safety, legality, efficiency, social expectations) when making driving decisions.

### Motivation

General-purpose VLMs (e.g., GPT-4V) can describe traffic scenes but are not reliably *sensitive to changes in normative reasons*. For example, a model might recommend overtaking a cyclist regardless of whether a double solid line is present. CARE-Drive exposes this insensitivity. This project addresses it by fine-tuning a VLM to internalize human-relevant reasoning structures.

---

## Framework: CARE-Drive

CARE-Drive evaluates VLMs across a structured decision-making pipeline:
```
Traffic Scene Image
        +
Structured Context (speed, distance, road markings, urgency)
        ↓
1. Who are the relevant human stakeholders? (cyclist, passenger, policymaker...)
2. What does each stakeholder expect the AV to do, and why?
3. What should the AV do, given all reasons?
        ↓
Evaluation: Does the decision change appropriately when contextual
            factors change? (sensitivity to reasons)
```

This fine-tuning project operationalises CARE-Drive as a training signal: the model is trained on (image, structured question) → answer pairs derived from CARE-Drive's multi-turn reasoning format.

---

## Project Objective

Fine-tune Qwen2-VL using QLoRA so that, given a traffic scene and a structured question about human stakeholders or normative expectations, the model produces responses that:

- correctly identify relevant human agents (on-road and off-road),
- accurately represent each agent's expectations and reasons,
- make final AV decisions that appropriately weigh those reasons, and
- change decisions when normative inputs change (the core CARE-Drive criterion).

---

## Project Status

| Component | Status |
|---|---|
| Dataset schema & sample data | ✅ Implemented |
| Message-format conversion (`dataset.py`) | ✅ Implemented |
| QLoRA training pipeline (`train.py`) | 🔄 In Progress |
| Inference script (`infer.py`) | ✅ Implemented |
| CARE-Drive evaluation harness | 🔄 In Progress |
| Full dataset collection & annotation | ⏳ Next Steps |
| Baseline vs. fine-tuned comparison | ⏳ Next Steps |
| Weights & results publication | ⏳ Next Steps |

---

## Technical Stack

| Component | Choice | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2-VL-7B-Instruct` | HuggingFace Hub |
| Fine-tuning method | SFT via TRL `SFTTrainer` | |
| Parameter efficiency | PEFT / LoRA | r=16, alpha=32 |
| Quantization | bitsandbytes 4-bit NF4 | QLoRA |
| Framework | HuggingFace Transformers + TRL | |
| Hardware target | 1× GPU, 16GB VRAM (e.g. A100 40GB for full run) | |

---

## Dataset Format

Each training sample is one `(image, question) → answer` pair in chat messages format. Multiple questions per image produce multiple independent samples.

**Raw annotation (`data/sample/sample_data.json`):**
```json
[
  {
    "image_path": "data/raw/scene_001.jpg",
    "context": {
      "speed_kmh": 42,
      "distance_to_front_m": 8.5,
      "road_marking": "double_solid",
      "oncoming_traffic": true,
      "weather": "clear"
    },
    "qa_pairs": [
      {
        "question": "Can you identify the human agents relevant to the AV in this scene?",
        "answer": "Yes. First, there is a cyclist directly ahead of the AV. Second, there is the passenger inside the AV."
      },
      {
        "question": "What does each agent expect the AV to do, and why?",
        "answer": "The cyclist may expect the AV to overtake, as remaining close behind may feel threatening. The passenger may also prefer overtaking due to time pressure. However, the policymaker expects the AV to stay behind, as overtaking across a double solid line is a legal violation."
      },
      {
        "question": "Given all human reasons, what should the AV do?",
        "answer": "The AV should stay behind the cyclist. Although the cyclist and passenger may prefer overtaking, the legal constraint imposed by the double solid line is a binding normative reason that overrides individual preferences in this context."
      }
    ]
  }
]
```

**Converted to training messages format:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an autonomous vehicle decision-making component designed to be responsive to human reasons. You will be given a traffic scene image and structured context data. Reason explicitly about human stakeholders and their expectations before making a decision."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<base64_or_path>"},
        {"type": "text", "text": "Context: speed=42 km/h, distance_to_front=8.5m, road_marking=double_solid, oncoming_traffic=true\n\nQuestion: Can you identify the human agents relevant to the AV in this scene?"}
      ]
    },
    {
      "role": "assistant",
      "content": "Yes. First, there is a cyclist directly ahead of the AV. Second, there is the passenger inside the AV."
    }
  ]
}
```

---

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/reason-driven-av.git
cd reason-driven-av
pip install -r requirements.txt

# Smoke-test dataset loading
python src/dataset.py

# Run inference with base model (no fine-tuning required)
python src/infer.py

# Launch fine-tuning (requires GPU)
python src/train.py --config configs/training.yaml
```

---

## Citation / Related Work

This project builds on the CARE-Drive evaluation framework. If you use or reference this work, please cite:
```
[Citation forthcoming — manuscript in preparation]
```

---

## License

MIT
