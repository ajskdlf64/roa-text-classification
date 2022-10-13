import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset, load_metric

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def main(args):

    # Header
    title = "TRAIN"
    str_out = "-" * 100 + "\n" + f"|| {title:^94} ||" + "\n" + "-" * 100
    print(str_out)

    # Args
    print("\nargs :", args, "\n")

    DATASET_LIST = os.listdir(Path("datasets/train"))
    DATASET_LIST = [file for file in DATASET_LIST if ".csv" in file]

    OUTPUTS = dict()

    for idx, FILE_NAME in enumerate(DATASET_LIST):

        TASK = FILE_NAME.split(".")[0]
        print(f"@@@@@@@@@@@@ [{idx+1}/{len(DATASET_LIST)}] {TASK}\n")

        dataset = load_dataset(
            "csv",
            data_files={
                "train": f"datasets/train/{FILE_NAME}",
                "test": f"datasets/val/{FILE_NAME}",
            },
        )

        metric = load_metric("f1")

        tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)

        def preprocess_function(examples):
            return tokenizer(examples["sentence"], truncation=True)

        encoded_dataset = dataset.map(preprocess_function, batched=True)

        label_list = ["True", "False"]
        num_labels = 2

        model = AutoModelForSequenceClassification.from_pretrained(
            args.backbone, num_labels=num_labels
        ).cuda()

        metric_name = "f1"

        model_name = args.backbone.split("/")[-1]

        train_args = TrainingArguments(
            f"models/{model_name}-finetuned-{TASK}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.max_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=False,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(
                predictions=predictions, references=labels, average="micro"
            )

        trainer = Trainer(
            model,
            train_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        OUTPUTS[TASK] = trainer.evaluate()

        print()

    for k,v in OUTPUTS.items() : 
        print(f"{k:25} | {v}")

    print("\nTrain Completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument("--seed", required=False, type=int, default=1234)
    parser.add_argument("--max_epochs", required=False, type=int, default=1)
    parser.add_argument("--lr", required=False, type=float, default=3e-5)
    parser.add_argument("--batch_size", required=False, type=int, default=16)
    parser.add_argument(
        "--backbone",
        required=False,
        type=str,
        default="distilbert-base-multilingual-cased",
    )
    args = parser.parse_args()
    main(args)
