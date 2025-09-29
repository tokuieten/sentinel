# train.py
import os
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert-imdb"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 256
SEED = 42

def preprocess(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def compute_metrics(pred):
    metric_acc = load_metric("accuracy")
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=preds, references=labels)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1) Load dataset
    raw = load_dataset("imdb")  # train/test with 25k each

    # 2) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3) Preprocess
    tokenized_train = raw["train"].map(lambda ex: preprocess(tokenizer, ex), batched=True)
    tokenized_test = raw["test"].map(lambda ex: preprocess(tokenizer, ex), batched=True)

    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in ["input_ids","attention_mask","label"]])
    tokenized_test = tokenized_test.remove_columns([c for c in tokenized_test.column_names if c not in ["input_ids","attention_mask","label"]])

    tokenized_train.set_format(type="torch")
    tokenized_test.set_format(type="torch")

    # 4) TrainingArguments & Trainer
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",           
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=SEED,
    fp16=torch.cuda.is_available()
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved model to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
