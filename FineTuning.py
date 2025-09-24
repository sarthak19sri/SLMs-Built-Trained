#  Fine-Tuning BERT-mini on Yelp Review Full (3-class Classification)

## 🔹 Dataset
- **Source**: [Yelp Review Full] from Hugging Face Datasets Hub.
- **Size**: ~650,000 reviews originally across 5 stars.
- **Label Mapping** (converted to 3 classes):
  - **Bad** → 1–2 stars
  - **Good** → 3 stars
  - **Excellent** → 4–5 stars
- This mapping transforms the dataset into a **3-class sentiment classification task**.

---

## 🔹 Model
- Base model: **BERT-mini** (`prajjwal1/bert-mini`), a compact transformer.
- Applied two fine-tuning strategies:
  - **Full Fine-Tuning** → All model parameters updated (All ~11M parameters updated.)
  - **LoRA (Low-Rank Adaptation)** →  Only ~0.1M (≈1%) parameters trained (attention projection adapters).(parameter-efficient).

---

## 🔹 Preprocessing
- Used the **BERT-mini tokenizer** for tokenization.
- Applied truncation/padding to fixed maximum sequence length.
- Generated **input IDs + attention masks** for model input.

---

## 🔹 Training Setup
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss Function**: Cross-Entropy Loss (multi-class classification)
- **Evaluation**: Accuracy, Precision, Recall, and F1-score measured after each epoch.

---
## 🔹 Results (after 5 epochs)
- **Full Fine-Tuning**:  
  - Accuracy: **72.1%**  
  - F1-score: **71.6%**  
  - Trainable parameters: **~11M**  
  - Training time: **~5–20 minutes on Colab T4 GPU**  

- **LoRA Fine-Tuning**:  
  - Accuracy: **60.4%**  
  - F1-score: **53.7%**  
  - Trainable parameters: **~0.1M (~1% of full)**  
  - Training time: **~4–15 minutes on Colab T4 GPU**
---

## 🔹 Key Insights
- Full fine-tuning **outperformed LoRA** in this setup.  
- LoRA, however, trained with **far fewer parameters**, showing a trade-off between efficiency and accuracy.
- This comparison highlights the effectiveness of **parameter-efficient fine-tuning techniques** vs **standard full fine-tuning**.
"""

# 1. Install dependencies
!pip install transformers datasets peft huggingface_hub

# 2. Login to Hugging Face Hub (enter your token when prompted)
# ==========================================================
from huggingface_hub import notebook_login
notebook_login()

# ========================
# 1. Install dependencies
# ========================
!pip install transformers datasets scikit-learn peft -q

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType

# 2. Load Yelp dataset
# ========================
dataset = load_dataset("yelp_review_full")

# Remap 5-star → 3 labels (bad/good/excellent)
def map_labels(example):
    if example["label"] in [0, 1]:  # 1-2 stars
        example["label"] = 0  # bad
    elif example["label"] == 2:     # 3 stars
        example["label"] = 1  # good
    else:                           # 4-5 stars
        example["label"] = 2  # excellent
    return example

dataset = dataset.map(map_labels)

# Sample smaller for fast training
train_ds = dataset["train"].shuffle(seed=42).select(range(5000))
eval_ds = dataset["test"].shuffle(seed=42).select(range(1000))

label2id = { "bad": 0, "good": 1, "excellent": 2 }
id2label = { 0: "bad", 1: "good", 2: "excellent" }

print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

# 3. Dataset class
# ========================
model_name = "prajjwal1/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 128

class SentimentDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": int(item["label"])
        }

train_dataset = SentimentDataset(train_ds, tokenizer)
eval_dataset = SentimentDataset(eval_ds, tokenizer)

# 4. Metrics
# ========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 5A. Full Fine-tuning
# ========================
model_full = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

training_args_full = TrainingArguments(
    output_dir="./bert-mini-full",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch", # Corrected argument name
    logging_dir="./logs_full",
    report_to="none"
)

trainer_full = Trainer(
    model=model_full,
    args=training_args_full,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer_full.train()
results_full = trainer_full.evaluate()
print("Full Fine-tuning:", results_full)

# Save model locally
output_dir_full = "bert-mini-yelp-3class-sarthak-full"
trainer_full.save_model(output_dir_full)
tokenizer.save_pretrained(output_dir_full)

# Push to Hugging Face Hub
model_full.push_to_hub("Sarthak1999/bert-mini-yelp-3class-sarthak-full")
tokenizer.push_to_hub("Sarthak1999/bert-mini-yelp-3class-sarthak-full")

# 5B. LoRA Fine-tuning
# ========================
model_lora = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model_lora = get_peft_model(model_lora, peft_config)

training_args_lora = TrainingArguments(
    output_dir="./bert-mini-lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    logging_dir="./logs_lora",
    report_to="none"
)

trainer_lora = Trainer(
    model=model_lora,
    args=training_args_lora,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer_lora.train()
results_lora = trainer_lora.evaluate()
print("LoRA Fine-tuning:", results_lora)

# ========================
# 6. Results
# ========================
bullet = (
    f"Evaluated full fine-tuning vs LoRA on BERT-mini"
    f"Acc. {results_full['eval_accuracy']*100:.1f}% vs {results_lora['eval_accuracy']*100:.1f}%; "
    f"F1 {results_full['eval_f1']*100:.1f}% vs {results_lora['eval_f1']*100:.1f}%."
)

print("\nOUTPUT:\n", bullet)

# Save model locally
output_dir_lora = "bert-mini-yelp-3class-sarthak-lora"
trainer_lora.save_model(output_dir_lora)
tokenizer.save_pretrained(output_dir_lora)

# Push to Hugging Face Hub
model_lora.push_to_hub("Sarthak1999/bert-mini-yelp-3class-sarthak-lora")
tokenizer.push_to_hub("Sarthak1999/bert-mini-yelp-3class-sarthak-lora")

# 7. Inference helper
# ========================
def predict_sentiment(text, model, tokenizer):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    # Move input tensors to the same device as the model
    device = model.device
    enc = {key: val.to(device) for key, val in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        pred = torch.argmax(out.logits, dim=-1).item()
    return id2label[pred]

print("Example:", predict_sentiment("the food was not delicious", model_full, tokenizer))
# What you'll get: # Corrected smart quote
