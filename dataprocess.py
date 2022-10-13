import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ORIGINAL_PATH = "datasets/original"


def main(args):

    # Header
    title = "DATAPROCESS"
    str_out = "-" * 100 + "\n" + f"|| {title:^94} ||" + "\n" + "-" * 100
    print(str_out)

    # Args
    print("\nargs :", args, "\n")

    # Target Dataset List
    DATASET_LIST = os.listdir(Path(ORIGINAL_PATH))

    # Load Dataset
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="[1/5] Load Dataset...",
    ):
        FILE_NAME = DATASET.split(".")[0]
        FILE_PATH = Path(ORIGINAL_PATH, DATASET)
        globals()[f"{FILE_NAME}"] = pd.read_excel(FILE_PATH, index_col=0)

    # Down Sampling
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="[2/5] Down Sampling...",
    ):
        FILE_NAME = DATASET.split(".")[0]

        globals()[f"{FILE_NAME}_0"] = globals()[f"{FILE_NAME}"][
            globals()[f"{FILE_NAME}"]["label"] == 0
        ]
        globals()[f"{FILE_NAME}_1"] = globals()[f"{FILE_NAME}"][
            globals()[f"{FILE_NAME}"]["label"] == 1
        ]

        globals()[f"{FILE_NAME}_0"] = globals()[f"{FILE_NAME}_0"].sample(
            len(globals()[f"{FILE_NAME}_1"])
        )
        globals()[f"{FILE_NAME}"] = pd.concat(
            [globals()[f"{FILE_NAME}_0"], globals()[f"{FILE_NAME}_1"]]
        )
        globals()[f"{FILE_NAME}"] = globals()[f"{FILE_NAME}"].sample(
            len(globals()[f"{FILE_NAME}"])
        )

        globals()[f"{FILE_NAME}"] = globals()[f"{FILE_NAME}"].dropna()
        globals()[f"{FILE_NAME}"] = globals()[f"{FILE_NAME}"].fillna("")

    # Text Preprocessing
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="[3/5] Preprocessing...",
    ):
        FILE_NAME = DATASET.split(".")[0]

        def preprocessing(title, text):
            return "[제목] " + title + " [본문]" + text

        globals()[f"{FILE_NAME}"]["sentence"] = globals()[f"{FILE_NAME}"][
            ["title", "text"]
        ].apply(lambda x: preprocessing(x["title"], x["text"]), axis=1)
        globals()[f"{FILE_NAME}"] = globals()[f"{FILE_NAME}"][["sentence", "label"]]

    # train,val,test split
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="[4/5] train,val,test split...",
    ):
        FILE_NAME = DATASET.split(".")[0]
        train, val, test = None, None, None
        total_data = len(globals()[f"{FILE_NAME}"])

        if args.test_ratio > 0.0:
            test_size = int(total_data * args.test_ratio)
            (
                globals()[f"{FILE_NAME}"],
                globals()[f"{FILE_NAME}_test"],
            ) = train_test_split(globals()[f"{FILE_NAME}"], test_size=test_size)

        test_size = int(total_data * args.val_ratio)
        (
            globals()[f"{FILE_NAME}_train"],
            globals()[f"{FILE_NAME}_val"],
        ) = train_test_split(globals()[f"{FILE_NAME}"], test_size=test_size)

    # Save File
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="[5/5] Save File...",
    ):
        FILE_NAME = DATASET.split(".")[0]
        globals()[f"{FILE_NAME}_train"].to_csv(
            f"datasets/train/{FILE_NAME}.csv", index=False
        )
        globals()[f"{FILE_NAME}_val"].to_csv(
            f"datasets/val/{FILE_NAME}.csv", index=False
        )
        globals()[f"{FILE_NAME}_test"].to_csv(
            f"datasets/test/{FILE_NAME}.csv", index=False
        )

    print("\nDatapocess Completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument("--seed", required=False, type=int, default=1234)
    parser.add_argument("--val_ratio", required=False, type=float, default=0.1)
    parser.add_argument("--test_ratio", required=False, type=float, default=0.1)
    args = parser.parse_args()
    main(args)
