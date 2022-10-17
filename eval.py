import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import pipeline

ORIGINAL_PATH = "datasets/original"


def main():

    # Header
    title = "EVAL"
    str_out = "-" * 100 + "\n" + f"|| {title:^94} ||" + "\n" + "-" * 100 + "\n"
    print(str_out)

    DATASET_LIST = os.listdir(Path(ORIGINAL_PATH))
    TASKS = [file.split(".")[0] for file in DATASET_LIST if ".csv" in file]

    OUTPUTS = list()

    for TASK in TASKS:

        str_out = "\n" + f"|| {TASK}" + "\n"

        test_df = pd.read_csv(f"datasets/test/{TASK}.csv")
        sentences = [sentence for sentence in test_df["sentence"]]

        classifier = pipeline(
            task="text-classification",
            model=f"models/distilbert-base-multilingual-cased-finetuned-{TASK}/",
            tokenizer=f"models/distilbert-base-multilingual-cased-finetuned-{TASK}/",
            device=0,
        )

        y_true = [label for label in test_df["label"]]
        y_pred = [
            int(classifier(x, truncation="only_first")[0]["label"].split("_")[1])
            for x in sentences
        ]

        target_names = ["class 0", "class 1"]

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=target_names)

        str_out = "\n" + f"@@@@@@@@@@@@@@ {TASK}" + "\n" * 2
        str_out += f"|| Acc : {acc:<18} ||" + "\n"
        str_out += f"|| pre : {pre:<18} ||" + "\n"
        str_out += f"|| rec : {rec:<18} ||" + "\n"
        str_out += f"|| f1  : {f1:<18} ||" + "\n"

        print(str_out)
        print(report)

        OUTPUT = {
            "TASK": TASK,
            "count": len(test_df),
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f1": f1,
        }
        OUTPUTS.append(OUTPUT)

    str_out = "\n" * 2 + "|| Summary" + "\n" + "-" * 110 + "\n"
    for OUTPUT in OUTPUTS:
        for k, v in OUTPUT.items():
            if type(v) == str:
                str_out += f"{k} : {v:23} | "
            else:
                v = round(v, 4)
                str_out += f"{k} : {v:<6} | "
        str_out += "\n"
    print(str_out)

    print("\nEval Completed.\n")


if __name__ == "__main__":
    main()
