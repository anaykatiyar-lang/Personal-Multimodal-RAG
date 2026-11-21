"""Supervised Fine-Tuning (SFT) pipeline for training custom 500M models on complex reasoning RAG data."""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReasoningChatDataset(Dataset):
    """Dataset converting ChatML reasoning turns to input_ids and labels for causal language modeling."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = ""
                for m in rec["messages"]:
                    text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels are input_ids with pad tokens masked to -100
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def run_sft_training(
    model_name_or_path: str,
    train_file: str,
    eval_file: Optional[str] = None,
    output_dir: str = "./models/fine_tuned_rag",
    num_train_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    device: str = "cpu",
    max_steps: Optional[int] = None
):
    """Execute supervised fine-tuning on complex reasoning dataset."""
    print("=" * 65, flush=True)
    print(f" Supervised Fine-Tuning Pipeline (Target: {model_name_or_path})", flush=True)
    print(f" Device: {device} | Epochs: {num_train_epochs}", flush=True)
    print("=" * 65, flush=True)

    print(f"Loading base model and tokenizer from {model_name_or_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    train_dataset = ReasoningChatDataset(train_file, tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()

    total_steps = len(train_loader) * num_train_epochs
    print(f" Training started: {len(train_dataset)} samples, {total_steps} total batches...", flush=True)

    step_count = 0
    running_loss = 0.0

    for epoch in range(num_train_epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            running_loss += outputs.loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 5 == 0 or step_count == 1:
                    avg_loss = running_loss / (5 * gradient_accumulation_steps if step_count > 1 else gradient_accumulation_steps)
                    print(f"  [Epoch {epoch+1}/{num_train_epochs}] Step {step_count}: Loss = {avg_loss:.4f}", flush=True)
                    running_loss = 0.0

            if max_steps and step_count >= max_steps:
                print(f" Reached target max_steps ({max_steps}). Finishing epoch early.", flush=True)
                break

    # Save fine-tuned weights and tokenizer
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f" Saving fine-tuned model weights to {out}...", flush=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    print("=" * 65, flush=True)
    print(" Fine-Tuning Completed Successfully! Weights ready for inference.", flush=True)
    print("=" * 65, flush=True)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RAG Reasoning Model")
    parser.add_argument("--model", type=str, default="./models/base_scratch_500m", help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSONL")
    parser.add_argument("--output_dir", type=str, default="./models/fine_tuned_rag", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or rocm")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps for demonstration or full run")
    args = parser.parse_args()

    run_sft_training(
        model_name_or_path=args.model,
        train_file=args.train_file,
        output_dir=args.output_dir,
        device=args.device,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps
    )
