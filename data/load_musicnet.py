# SYSTEM IMPORTS
from typing import Dict, List, Tuple
import argparse
import numpy as np
import os
import requests


# PYTHON PROJECT IMPORTS



# CONSTANTS
FILENAME: str = "point_cloud.npy"


def load() -> np.ndarray:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="the directory to load data from")
    args = parser.parse_args()

    filepath: str = os.path.join(args.data_dir, FILENAME)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(filepath):
        raise ValueError("ERROR: file [%s] does not exist at path [%s]" % (FILENAME, args.data_dir))
    return  np.load(filepath)


if __name__ == "__main__":
    X: np.ndarray = load()
    print(X.shape, X.dtype)

