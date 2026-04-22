import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- TPU SPECIFIC IMPORTS ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("❌ torch_xla not found.")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
NUM_LABELS = 6
DEVICE = torch_xla.device()

LABEL_MAP = {
    "remember": 0, "understand": 1, "apply": 2,
    "analyze": 3, "evaluate": 4, "create": 5
}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("/content/drive/MyDrive/colab/data/obe_dataset.csv")

# Clean data
df = df.dropna(subset=["question", "bloom_level"])
df["bloom_level"] = df["bloom_level"].str.lower().str.strip()
df = df[df["bloom_level"].isin(LABEL_MAP)]
df["label"] = df["bloom_level"].map(LABEL_MAP)

# -----------------------------
# LDL TARGET (Calculated BEFORE balancing to avoid KeyError)
# -----------------------------
def label_to_distribution(y, num_classes=6, sigma=1.0):
    dist = np.zeros(num_classes)
    for i in range(num_classes):
        dist[i] = np.exp(-((i - y) ** 2) / (2 * sigma ** 2))
    return dist / dist.sum()

df["dist"] = df["label"].apply(label_to_distribution)

# -----------------------------
# BALANCE
# -----------------------------
# FIX: Removed include_groups=False so 'label' stays in the dataframe
df = df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 3000), random_state=42)
)
print("✅ Dataset size:", len(df))
print("📊 Label distribution:\n", df["label"].value_counts())

# -----------------------------
# INPUT
# -----------------------------
def build_input(row):
    return f"Subject: {row['subject']} Topic: {row['topic']} Question: {row['question']}"

df["text"] = df.apply(build_input, axis=1)

# -----------------------------
# SPLIT & DATASET
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("dist", "labels")
val_ds = val_ds.rename_column("dist", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "label"])

# -----------------------------
# MODEL
# -----------------------------
class LDLModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        log_probs = torch.log_softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss_fn = nn.KLDivLoss(reduction="batchmean")
            loss = loss_fn(log_probs.float(), labels.float())
        return {"loss": loss, "logits": logits}

model = LDLModel(MODEL_NAME, NUM_LABELS).to(DEVICE)

# -----------------------------
# TRAINING CONFIG
# -----------------------------
training_args = TrainingArguments(
    output_dir="/content/ldl_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    optim="adamw_torch", # Essential for TPU compatibility
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)
    return {"accuracy": accuracy_score(true, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# -----------------------------
# TRAIN
# -----------------------------
print("🚀 Training LDL model on TPU...")
trainer.train()

# -----------------------------
# UNCERTAINTY & EVALUATION
# -----------------------------
def mc_dropout_predict(model, input_ids, attention_mask, T=10):
    model.train()
    preds = []
    for _ in range(T):
        with torch.no_grad():
            logits = model(input_ids, attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds.append(probs.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    return preds.mean(dim=0), preds.var(dim=0)

def compute_uncertainty(probs, variance):
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    epistemic = torch.mean(variance, dim=-1)
    return entropy, epistemic

def reject_decision(probs, entropy, epistemic, conf_th=0.6, ent_th=1.5, epi_th=0.02):
    confidence = torch.max(probs, dim=-1).values
    reject = (confidence < conf_th) | (entropy > ent_th) | (epistemic > epi_th)
    return reject, confidence

def evaluate_with_rejection(model, dataset):
    model.eval()
    total, accepted, correct = 0, 0, 0
    print("🔍 Starting TPU Evaluation...")
    for sample in dataset:
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(DEVICE)
        probs, var = mc_dropout_predict(model, input_ids, attention_mask)
        probs, var = probs.cpu(), var.cpu()
        entropy, epi = compute_uncertainty(probs, var)
        reject, conf = reject_decision(probs, entropy, epi)
        pred = torch.argmax(probs, dim=-1).item()
        true = sample["label"]
        total += 1
        if not reject.item():
            accepted += 1
            if pred == true:
                correct += 1

    cov = accepted/total if total > 0 else 0
    acc = correct/accepted if accepted > 0 else 0
    print(f"\n📊 Coverage: {cov:.4f}")
    print(f"🎯 Accuracy (accepted): {acc:.4f}")

evaluate_with_rejection(model, val_ds)

# Save using XLA-safe method
import torch_xla.core.xla_model as xm
xm.save(model.state_dict(), "/content/ldl_uncertainty_model.pt")
print("✅ FULL SYSTEM COMPLETE ON TPU")