# SYSTEM IMPORTS
from typing import Dict, List, Tuple
import argparse
import numpy as np
import os
import requests


# PYTHON PROJECT IMPORTS



# CONSTANTS
FILENAME: str = "point_cloud.npy"


def load(data_path: str) -> np.ndarray:
    if not os.path.exists(data_path):
        raise ValueError("ERROR: file [%s] does not exist" % data_path)
    return  np.load(data_path)


def main() -> np.ndarray:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="the directory to load data from")
    args = parser.parse_args()

    filepath: str = os.path.join(args.data_dir, FILENAME)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    return load(filepath)


if __name__ == "__main__":
    X: np.ndarray = main()
    print(X.shape, X.dtype)

