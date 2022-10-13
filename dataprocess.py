import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ORIGINAL_PATH = "datasets/original"


def main(args):

    # Header
    title = "DATAPROCESS"
    str_out = "-" * 60 + "\n" + f"|| {title:^54} ||" + "\n" + "-" * 60
    print(str_out)

    # Target Dataset List
    DATASET_LIST = os.listdir(Path(ORIGINAL_PATH))

    # Load Dataset
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="Load Dataset...",
    ):
        FILE_NAME = DATASET.split(".")[0]
        FILE_PATH = Path(ORIGINAL_PATH, DATASET)
        globals()[f"{FILE_NAME}"] = pd.read_excel(FILE_PATH)

    # Down Sampling
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="Load Dataset...",
    ):
        pass

    # Text Preprocessing
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="Load Dataset...",
    ):
        pass

    # Save File
    for DATASET in tqdm(
        DATASET_LIST,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        total=len(DATASET_LIST),
        desc="Load Dataset...",
    ):
        pass

    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument("--seed", required=False, type=int, default=1234)
    parser.add_argument("--val_ratio", required=False, type=float, default=0.1)
    parser.add_argument("--make_test_ratio", required=False, type=float, default=0.1)
    args = parser.parse_args()
    main(args)
