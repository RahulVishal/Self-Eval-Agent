"""
Transformer fine-tune for the depression-detection project: DistilBERT.

Run after data_pipeline.py has produced train.csv / val.csv / test.csv.
Requires: transformers, datasets, torch, scikit-learn

    pip install transformers datasets torch scikit-learn accelerate
    python transformer_finetune.py
"""

import numpy as np
import pandas as pd
from datasets import Dataset  # requires: pip install datasets
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

TEXT_COL = "clean_text"
LABEL_COL = "is_depression"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
OUTPUT_DIR = "distilbert_depression"


def load_splits():
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, val_df, test_df


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "text", LABEL_COL: "label"}),
        preserve_index=False,
    )


def tokenize_fn(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }


def main():
    train_df, val_df, test_df = load_splits()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = to_hf_dataset(train_df).map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    val_ds = to_hf_dataset(val_df).map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    test_ds = to_hf_dataset(test_df).map(lambda b: tokenize_fn(b, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("\n=== Validation (best checkpoint) ===")
    print(trainer.evaluate(val_ds))

    print("\n=== Test ===")
    print(trainer.evaluate(test_ds))

    trainer.save_model(f"{OUTPUT_DIR}/best")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")
    print(f"\nSaved fine-tuned model to {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()