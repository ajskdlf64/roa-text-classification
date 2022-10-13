import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():

    # Header
    title = "EVAL"
    str_out = "-" * 100 + "\n" + f"|| {title:^94} ||" + "\n" + "-" * 100
    print(str_out)

    print("\nEval Completed.\n")


if __name__ == "__main__":
    main()
