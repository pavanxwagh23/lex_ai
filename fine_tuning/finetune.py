from __future__ import annotations
import json, os, sys
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH  = PROJECT_ROOT / "fine_tuning" / "data" / "legal_qa_dataset.jsonl"
OUTPUT_DIR    = PROJECT_ROOT / "models" / "lex_ai_custom_llm"
BASE_MODEL    = "microsoft/Phi-3-mini-4k-instruct"

# RTX 3050 6GB VRAM — SPEED-OPTIMISED for 8-hour target
EPOCHS        = 3
BATCH_SIZE    = 1       # 1 per device — safe for 6GB VRAM
GRAD_ACCUM    = 8       # Effective batch = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 256     # Reduced 512→256: ~2x speedup (legal Q&A avg ~150 tokens, fits fine)
MAX_SAMPLES   = 10000   # Use 10K of 27K samples: mathematically guarantees <8 hour completion
LORA_RANK     = 8       # Reduced 16→8: ~10% faster backward pass
LORA_ALPHA    = 16      # Keep 2× LoRA rank ratio
GRAD_CKPT     = False   # Disabled: Trade the empty 2.5GB VRAM for raw speed!
ALLOW_DIRTY_DATASET = os.getenv("ALLOW_DIRTY_FINETUNE", "false").lower() in {"1", "true", "yes", "on"}



def check_deps():
    missing = []
    for pkg in ["transformers", "peft", "datasets", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Install: pip install " + " ".join(missing))
        return False
    return True


def load_data():
    from datasets import Dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Run generate_dataset.py first.")
    samples = []
    skipped = 0
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                p = obj.get("prompt", "")
                c = obj.get("completion", "")
                if isinstance(c, dict):
                    c = json.dumps(c, ensure_ascii=False)
                if p and c:
                    samples.append({"prompt": str(p), "completion": str(c)})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    print("Loaded " + str(len(samples)) + " valid samples. Malformed rows: " + str(skipped))

    # Subsample for speed while keeping diversity
    if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
        import random
        random.seed(42)  # Reproducible shuffle
        random.shuffle(samples)
        samples = samples[:MAX_SAMPLES]
        print("Subsampled to " + str(MAX_SAMPLES) + " samples for speed target.")

    return Dataset.from_list(samples)


def format_prompt(example):
    nl = "\n"
    text = "### Instruction:" + nl + example["prompt"] + nl + nl + "### Response:" + nl + example["completion"] + nl
    return {"text": text}


def finetune():
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print("=" * 60)
    print("  Lex AI Legal Model Fine-Tuning")
    print("  Base  : " + BASE_MODEL)
    print("  Output: " + str(OUTPUT_DIR))
    print("=" * 60)

    if not check_deps():
        sys.exit(1)

    from fine_tuning.validate_dataset import validate_dataset

    issues = validate_dataset(DATASET_PATH)
    if issues and not ALLOW_DIRTY_DATASET:
        print("\nDataset validation failed: " + str(len(issues)) + " issue(s) found.")
        print("Fix the dataset before training, or set ALLOW_DIRTY_FINETUNE=true to override.")
        print("First issues:")
        for issue in issues[:20]:
            print(f"  line {issue.line}: {issue.code}: {issue.message}")
        sys.exit(1)
    if issues:
        print("\nWARNING: training with " + str(len(issues)) + " dataset validation issue(s).")

    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    print("\nLoading base model in 4-bit precision (~2GB VRAM)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    print("GPU available: " + str(use_gpu))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if use_gpu else None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"} if use_gpu else "auto",
        trust_remote_code=True,
    )
    if use_gpu:
        model = prepare_model_for_kbit_training(model)

    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: " + str(n_params))

    # Enable gradient checkpointing BEFORE applying LoRA — saves ~40% VRAM
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Apply LoRA adapters — trains only 0.5% of weights
    # Phi-3 uses qkv_proj / o_proj / gate_up_proj / down_proj (NOT q_proj/v_proj like LLaMA)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()


    # Prepare and tokenize dataset
    dataset = load_data().map(format_prompt)

    def tok_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            # No padding here — DataCollator handles dynamic per-batch padding (faster)
        )

    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        fp16=use_gpu,
        bf16=False,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        group_by_length=True,      # Batch similar-length sequences together — reduces padding waste
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer  = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("\nStarting fine-tuning (Resuming from checkpoint if available)...")
    trainer.train(resume_from_checkpoint=True)

    print("\nSaving model to: " + str(OUTPUT_DIR))
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("\nFine-tuning complete!")
    print("Serve your model: python fine_tuning/serve.py")


if __name__ == "__main__":
    finetune()
