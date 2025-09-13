import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load dataset from raw folder
df = pd.read_csv("data/raw/geeta_dataset.csv")

# Prepare instruction-response pairs
records = []
for _, row in df.iterrows():
    instruction = f"Explain Chapter {row['chapter']} Verse {row['verse']} in English"
    response = row['english']
    records.append({"instruction": instruction, "output": response})

# Split 80% train, 20% test
train_data, test_data = train_test_split(records, test_size=0.2, random_state=42)

# Save JSONL in processed folder
with open("data/processed/gita_train.jsonl", "w") as f:
    for rec in train_data:
        f.write(json.dumps(rec) + "\n")

with open("data/processed/gita_test.jsonl", "w") as f:
    for rec in test_data:
        f.write(json.dumps(rec) + "\n")

print("✅ Preprocessing complete: gita_train.jsonl and gita_test.jsonl created")
