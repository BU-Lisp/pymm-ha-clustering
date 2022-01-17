# SYSTEM IMPORTS
from typing import Dict, List, Tuple
import argparse
import numpy as np
import os
import requests


# PYTHON PROJECT IMPORTS



# CONSTANTS
FILENAME: str = "mnist.npz"


def load() -> np.ndarray:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="the directory to load data from")
    args = parser.parse_args()

    mnist_filepath: str = os.path.join(args.data_dir, FILENAME)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(mnist_filepath):
        raise ValueError("ERROR: file [%s] does not exist at path [%s]" % (FILENAME, args.data_dir))
    data_dict: Dict[str, np.ndarray] = np.load(mnist_filepath)
    X: np.ndarray = np.vstack([data_dict["X_train"], data_dict["X_test"]])
    return X.reshape(X.shape[0], -1)


if __name__ == "__main__":
    X: np.ndarray = load()
    print(X.shape, X.dtype)

