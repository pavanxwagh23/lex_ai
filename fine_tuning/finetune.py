from __future__ import annotations
import json, sys
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH  = PROJECT_ROOT / 'fine_tuning' / 'data' / 'legal_qa_dataset.jsonl'
OUTPUT_DIR    = PROJECT_ROOT / 'models' / 'lex_ai_custom_llm'
BASE_MODEL    = 'microsoft/Phi-3-mini-4k-instruct'

EPOCHS        = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 1024
LORA_RANK     = 16
LORA_ALPHA    = 32


def check_deps():
    missing = []
    for pkg in ['transformers', 'peft', 'datasets', 'accelerate']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print('Install: pip install ' + ' '.join(missing))
        return False
    return True


def load_data():
    from datasets import Dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError('Run generate_dataset.py first.')
    samples, skipped = [], 0
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                p = obj.get('prompt', '')
                c = obj.get('completion', '')
                if isinstance(c, dict):
                    c = json.dumps(c, ensure_ascii=False)
                if p and c:
                    samples.append({'prompt': str(p), 'completion': str(c)})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    print('Loaded ' + str(len(samples)) + ' samples (skipped ' + str(skipped) + ')')
    return Dataset.from_list(samples)


def format_prompt(example):
    text = '### Instruction:\n' + example['prompt'] + '\n\n### Response:\n' + example['completion'] + '\n'
    return {'text': text}


def finetune():
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print('=' * 60)
    print('  Lex AI Legal Model Fine-Tuning')
    print('  Base  : ' + BASE_MODEL)
    print('  Output: ' + str(OUTPUT_DIR))
    print('=' * 60)

    if not check_deps():
        sys.exit(1)

    print('\nLoading base model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ' + str(n_params))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'v_proj'],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset  = load_data().map(format_prompt)
    def tok_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_fp16 = torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy='epoch',
        fp16=use_fp16,
        report_to='none',
        dataloader_pin_memory=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer  = Trainer(
        model=model, args=train_args,
        train_dataset=tokenized, data_collator=collator,
    )

    print('\nStarting fine-tuning...')
    trainer.train()

    print('\nSaving model to ' + str(OUTPUT_DIR))
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print('\nDone! Run: python fine_tuning/serve.py to serve your model.')


if __name__ == '__main__':
    finetune()
