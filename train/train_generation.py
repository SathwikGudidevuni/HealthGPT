# Training script for generation model
"""
Fine-tune a seq2seq generation model (T5/Flan-T5) on vlhealth dataset.

Expected dataset format:
  CSV/JSON with columns: input_text,target_text

Usage examples:
  python train/train_generation.py --train_csv data/gen_train.csv --val_csv data/gen_val.csv --output_dir outputs/gen_t5
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=None, help="HF dataset id (optional)")
    parser.add_argument("--train_csv", type=str, default=None, help="Local CSV train file path")
    parser.add_argument("--val_csv", type=str, default=None, help="Local CSV val file path")
    parser.add_argument("--input_field", type=str, default="input_text")
    parser.add_argument("--target_field", type=str, default="target_text")
    parser.add_argument("--model_name", type=str, default=os.environ.get("BASE_GEN_MODEL", "t5-small"))
    parser.add_argument("--output_dir", type=str, default="outputs/gen_t5")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    def preprocess(batch):
        inputs = tokenizer(batch[args.input_field], truncation=True, padding="max_length", max_length=max_input_length)
        targets = tokenizer(batch[args.target_field], truncation=True, padding="max_length", max_length=max_target_length)
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_tokenized = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tokenized = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names) if val_ds is not None else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch" if eval_tokenized is not None else "no",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=args.num_train_epochs,
        fp16=False,
        load_best_model_at_end=True if eval_tokenized is not None else False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved generator to: {args.output_dir}")


if __name__ == "__main__":
    main()
