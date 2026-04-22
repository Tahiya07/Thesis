#scripts/train_coral_colab.py

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------

MODEL_NAME = "microsoft/deberta-v3-base"   # stable + strong baseline
NUM_LABELS = 6
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3

LABELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/obe_dataset.csv")

# Normalize labels
df["bloom_level"] = (
    df["bloom_level"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Map to numeric
df["labels"] = df["bloom_level"].map(label2id)

# Drop invalid rows
df = df.dropna(subset=["question", "labels"])
df["labels"] = df["labels"].astype(int)

print(f"✅ Dataset size: {len(df)}")

# -----------------------------
# STRATIFIED SAMPLING (BALANCE)
# -----------------------------

df = df.groupby("labels", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 3000), random_state=42)
)

print("📊 Label distribution:")
print(df["labels"].value_counts())

# -----------------------------
# TRAIN / VAL SPLIT
# -----------------------------

train_df, val_df = train_test_split(
    df, test_size=0.1, stratify=df["labels"], random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# -----------------------------
# TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------------
# MODEL (4-bit NOT needed here since DeBERTa is smaller)
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# -----------------------------
# LORA (LIGHTWEIGHT FINETUNING)
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_proj", "value_proj"],  # correct for DeBERTa
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(model, lora_config)

print("✅ LoRA applied")

# -----------------------------
# METRICS
# -----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# -----------------------------
# TRAINING ARGS
# -----------------------------

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,

    fp16=False,   
    bf16=False,  

    gradient_checkpointing=True,
    report_to="none",
)
# -----------------------------
# TRAINER
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# TRAIN
# -----------------------------

print("🚀 Starting training...")
trainer.train()

# -----------------------------
# SAVE
# -----------------------------

trainer.save_model("models/coral_bloom")
tokenizer.save_pretrained("models/coral_bloom")

print("✅ Training complete and model saved!")