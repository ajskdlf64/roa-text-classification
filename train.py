import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main(args):

    # Header
    title = "TRAIN"
    str_out = "-" * 100 + "\n" + f"|| {title:^94} ||" + "\n" + "-" * 100
    print(str_out)

    # Args
    print("\nargs :", args, "\n")

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
