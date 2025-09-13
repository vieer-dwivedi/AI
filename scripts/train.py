# gita_finetune.py
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os

# -----------------------------
# 1. Load CSV and preprocess
# -----------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/raw/geeta_dataset.csv")
records = []

for _, row in df.iterrows():
    instruction = f"Explain Chapter {row['chapter']} Verse {row['verse']} in English"
    response = row['english']
    records.append({"instruction": instruction, "output": response})

# -----------------------------
# 2. Train/test split
# -----------------------------
train_data, test_data = train_test_split(records, test_size=0.2, random_state=42)

# Save JSONL
with open("data/raw/gita_train.jsonl", "w") as f:
    for rec in train_data:
        f.write(json.dumps(rec) + "\n")

with open("data/raw/gita_test.jsonl", "w") as f:
    for rec in test_data:
        f.write(json.dumps(rec) + "\n")

print("✅ Preprocessing complete: gita_train.jsonl and gita_test.jsonl created")

# -----------------------------
# 3. Load dataset for training
# -----------------------------
train_dataset = load_dataset("json", data_files="data/processed/gita_train.jsonl")['train']
test_dataset = load_dataset("json", data_files="data/processed/gita_test.jsonl")['train']

# -----------------------------
# 4. Load tokenizer and model
# -----------------------------
model_name = "openbmb/MiniCPM4.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# 5. Tokenize dataset
# -----------------------------
def tokenize_fn(example):
    # Concatenate instruction + output as input for causal LM
    return tokenizer(example["instruction"] + " " + example["output"], truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset  = test_dataset.map(tokenize_fn, batched=True)

# -----------------------------
# 6. Configure LoRA
# -----------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 7. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/gita_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    evaluation_strategy="steps",
    save_total_limit=2,
    fp16=True
)

# -----------------------------
# 8. Trainer & training
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# -----------------------------
# 9. Save model & tokenizer
# -----------------------------
model.save_pretrained("models/gita_finetuned")
tokenizer.save_pretrained("models/gita_finetuned")
print("🎉 Fine-tuning complete! Model saved in models/gita_finetuned")
