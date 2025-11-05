# Training script for comprehension model
"""
Fine-tune a classifier for the comprehension task (intent classification) on the vlhealth dataset.

Usage examples:
  # Using a local CSV dataset with columns: text,label
  python train/train_comprehension.py --train_csv data/comp_train.csv --val_csv data/comp_val.csv --output_dir outputs/comp_classifier

  # Using an HF dataset id (if available)
  python train/train_comprehension.py --dataset_id owner/vlhealth --split_train train --split_val validation --text_field text --label_field label --output_dir outputs/comp_classifier
"""
import argparse
import os
import json
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=None, help="HF dataset id (optional)")
    parser.add_argument("--train_csv", type=str, default=None, help="Local CSV train file path")
    parser.add_argument("--val_csv", type=str, default=None, help="Local CSV val file path")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--model_name", type=str, default=os.environ.get("BASE_COMP_MODEL", "bert-base-uncased"))
    parser.add_argument("--output_dir", type=str, default="outputs/comp_classifier")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()


def load_data(args):
    if args.dataset_id:
        ds = load_dataset(args.dataset_id)
        train = ds["train"]
        val = ds.get("validation") or ds.get("dev") or None
    else:
        data_files = {}
        if args.train_csv:
            data_files["train"] = args.train_csv
        if args.val_csv:
            data_files["validation"] = args.val_csv
        if not data_files:
            raise ValueError("Provide --dataset_id or at least --train_csv")
        ds = load_dataset("csv", data_files=data_files)
        train = ds["train"]
        val = ds.get("validation")
    return train, val


def main():
    args = parse_args()
    train_ds, val_ds = load_data(args)

    # collect labels and create mapping
    labels = sorted(list(set(train_ds[args.label_field])))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(batch):
        toks = tokenizer(batch[args.text_field], truncation=True, padding="max_length", max_length=args.max_length)
        toks["labels"] = [label2id[l] for l in batch[args.label_field]]
        return toks

    train_tokenized = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tokenized = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names) if val_ds is not None else None

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_true = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels_true)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if eval_tokenized is not None else "no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        fp16=False,
        load_best_model_at_end=True if eval_tokenized is not None else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_tokenized is not None else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Save metadata (label mapping)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(args.output_dir, "id2label.json"), "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f)

    print(f"Saved classifier and mappings to {args.output_dir}")


if __name__ == "__main__":
    main()
