"""
fine_tuning/finetune.py
========================
Fine-tunes a small open-source base model on your legal Q&A dataset
to produce YOUR OWN custom legal AI model.

Base model: microsoft/phi-2 (2.7B parameters — fast to train, runs on 8GB RAM)
Method: LoRA (Low-Rank Adaptation) via PEFT — trains only 0.1% of weights

Output: models/lex_ai_custom_llm/  (fully owned by you)

Requirements (install first):
    pip install transformers peft datasets accelerate bitsandbytes

Usage:
    python fine_tuning/finetune.py

Training time estimate:
    - CPU only: ~2-4 hours (not recommended)
    - NVIDIA GPU (8GB VRAM): ~15-30 minutes
    - NVIDIA GPU (16GB+ VRAM): ~8-15 minutes
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "fine_tuning" / "data" / "legal_qa_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "lex_ai_custom_llm"
BASE_MODEL = "microsoft/phi-2"   # Swap to "microsoft/phi-3-mini-4k-instruct" for smarter model

# Fine-tuning hyperparameters
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32


def check_dependencies() -> bool:
    missing = []
    for pkg in ["transformers", "peft", "datasets", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    return True


def load_dataset():
    from datasets import load_dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run: python fine_tuning/generate_dataset.py"
        )
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    print(f"✅ Loaded {len(dataset)} training samples")
    return dataset


def format_prompt(example: dict) -> dict:
    """Format prompt+completion into instruction-tuning format."""
    text = (
        f"### Instruction:\n{example['prompt']}\n\n"
        f"### Response:\n{example['completion']}\n"
    )
    return {"text": text}


def finetune():
    print("=" * 60)
    print("  Lex AI — Custom Legal LLM Fine-Tuning")
    print(f"  Base Model : {BASE_MODEL}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    # 1. Load tokenizer & base model
    print(f"\n📥 Loading base model: {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 2. Apply LoRA — only train a small adapter on top of frozen weights
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Prepare dataset
    raw_dataset = load_dataset()
    formatted = raw_dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )

    tokenized_dataset = formatted.map(tokenize, batched=True, remove_columns=formatted.column_names)

    # 4. Training configuration
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 5. Train!
    print("\n🚀 Starting fine-tuning...")
    trainer.train()

    # 6. Save the model
    print(f"\n💾 Saving model to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("  ✅ Fine-tuning complete!")
    print(f"  Your personal legal AI model is saved at:")
    print(f"  {OUTPUT_DIR}")
    print("\n  To serve it, run: python fine_tuning/serve.py")
    print("=" * 60)


if __name__ == "__main__":
    finetune()
