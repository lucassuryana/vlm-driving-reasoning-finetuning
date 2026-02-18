"""
train.py
QLoRA fine-tuning of Qwen2-VL using HuggingFace TRL SFTTrainer.
Targets 16GB VRAM via 4-bit NF4 quantization + gradient checkpointing.

Usage:
    python src/train.py --config configs/training.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
from datasets import Dataset
import json

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Qwen2-VL requires its own model class
from qwen_vl_utils import process_vision_info  # from qwen-vl-utils pip package
from transformers import Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Collator — converts message dicts to model inputs
# ---------------------------------------------------------------------------

class Qwen2VLCollator:
    """
    Applies the Qwen2-VL processor to a batch of message-format samples.
    Produces input_ids, attention_mask, pixel_values, and labels.
    Labels mask all tokens except the assistant turn.
    """

    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts, image_inputs_list = [], []

        for sample in batch:
            messages = sample["messages"]
            # Build the prompt text via the processor's chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            # Extract vision inputs (handles base64 / path / URL)
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        inputs = self.processor(
            text=texts,
            images=image_inputs_list if any(image_inputs_list) else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build labels: mask everything except assistant response tokens
        # TODO: implement precise assistant-token masking using chat template offsets
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(cfg: dict):
    model_name = cfg["model"]["name"]
    output_dir = cfg["training"]["output_dir"]
    data_path = cfg["data"]["train_path"]

    # -- 4-bit quantization config --
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading model: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing

    # -- LoRA config --
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -- Processor --
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # -- Dataset --
    raw = load_jsonl(data_path)
    dataset = Dataset.from_list(raw)
    collator = Qwen2VLCollator(processor, max_length=cfg["data"].get("max_length", 1024))

    # -- Training arguments --
    tr = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr.get("epochs", 3),
        per_device_train_batch_size=tr.get("batch_size", 1),
        gradient_accumulation_steps=tr.get("grad_accum", 8),
        learning_rate=tr.get("lr", 2e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=tr.get("save_steps", 100),
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",  # set to "wandb" when ready
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # -- Trainer --
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        # TODO: add eval_dataset and compute_metrics for CARE-Drive eval
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
