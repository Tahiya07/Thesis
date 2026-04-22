#scripts/train_coral_no_lora.py

import os
import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
NUM_LABELS = 6  # Bloom levels
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LABEL MAPPING (ORDERED!)
# -----------------------------
LABEL_MAP = {
    "remember": 0,
    "understand": 1,
    "apply": 2,
    "analyze": 3,
    "evaluate": 4,
    "create": 5
}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("/content/drive/MyDrive/colab/data/obe_dataset.csv")

df = df.dropna(subset=["question", "bloom_level"])

df["bloom_level"] = df["bloom_level"].str.lower().str.strip()
df = df[df["bloom_level"].isin(LABEL_MAP.keys())]

df["labels"] = df["bloom_level"].map(LABEL_MAP)

print("✅ Dataset size:", len(df))

# -----------------------------
# OPTIONAL: BALANCED SAMPLING
# -----------------------------
df = df.groupby("labels", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 3000), random_state=42)
)

print("📊 Label distribution:")
print(df["labels"].value_counts())


# -----------------------------
# INPUT CONSTRUCTION (IMPORTANT)
# -----------------------------
def build_input(row):
    return f"""
Subject: {row['subject']}
Topic: {row['topic']}
Cognitive Skill: {row['cognitive_skill']}

Question: {row['question']}
"""


df["text"] = df.apply(build_input, axis=1)

# -----------------------------
# TRAIN/VAL SPLIT
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["labels"],
    random_state=42
)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )


train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# -----------------------------
# CORAL MODEL
# -----------------------------
class CoralModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = torch.nn.Linear(hidden_size, num_labels - 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        if labels is not None:
            # Convert labels to ordinal targets
            batch_size = labels.size(0)
            ordinal_labels = torch.zeros((batch_size, NUM_LABELS - 1)).to(labels.device)

            for i in range(NUM_LABELS - 1):
                ordinal_labels[:, i] = (labels > i).float()

            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, ordinal_labels)

            return {"loss": loss, "logits": logits}

        return {"logits": logits}


model = CoralModel(MODEL_NAME, NUM_LABELS).to(DEVICE)


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    preds = (probs > 0.5).sum(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds)
    }


# -----------------------------
# TRAINING CONFIG
# -----------------------------
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/colab/results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,  # IMPORTANT FIX
    bf16=False,
    gradient_checkpointing=True,
    report_to="none"
)

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# -----------------------------
# TRAIN
# -----------------------------
print("🚀 Starting CORAL training...")
trainer.train()

# -----------------------------
# SAVE
# -----------------------------
trainer.save_model("/content/drive/MyDrive/colab/coral_no_lora")
print("✅ CORAL model saved!")