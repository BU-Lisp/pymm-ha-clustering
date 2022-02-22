import sys, getopt
import os
import argparse
from datetime import datetime

import numpy as np

# Project imports
from algs.kmeans import KMeans
from data.load_mnist import load as load_mnist
from data.load_imagenet2012 import load as load_imagenet
from data.load_musicnet import load as load_musicnet

# Custom dataset names
MNIST = "mnist.npz"
MUSICNET = "point_cloud.npy"
IMAGENET = ["imagenet_500k.npy", "imagenet2012_800k.npy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("--num-clusters", type=int, required=True)
    args = parser.parse_args()

    a = datetime.now()

    NUM_CLUSTERS: int = args.num_clusters

    FILEPATH: str = args.dataset_path
    if not os.path.exists(FILEPATH):
        os.makedirs(FILEPATH)
    FILENAME: str = args.dataset_path.rsplit('/', 1)[-1]
    FILEPATH2: str = FILEPATH[:-len(FILENAME)]
    
    pretty_print("DATASET:", FILENAME)
    pretty_print("# CLUSTERS (K):", NUM_CLUSTERS)
    X, data_load_time = populate_ndarray(FILENAME, FILEPATH)
    pretty_print("DATA_LOADING_TIME:", data_load_time)
    
    m, X, training_time = exec(X, NUM_CLUSTERS)
    pretty_print("MODEL_TRAINING_TIME:", training_time)
    pretty_print("MODEL_COST:", m.cost(X))
    print(m.centers)

    t_delta = datetime.now() - a
    pretty_print("TOTAL_SCRIPT_RUNTIME:", t_delta)

# =======================================================

def populate_ndarray(FILENAME: str, FILEPATH: str):
    """Leverage load_xyz method for corresponding dataset"""
    if FILENAME == MNIST:
        X, t_delta = wrap_loader(load_mnist, FILEPATH)
        # X = load_mnist(FILEPATH)
    elif FILENAME in IMAGENET:
        X, t_delta = wrap_loader(load_imagenet, FILEPATH)
        # X = load_imagenet(FILEPATH)
    elif FILENAME == MUSICNET:
        X, t_delta = wrap_loader(load_musicnet, FILEPATH)
        # X = load_musicnet(FILEPATH)
    else:
        raise ValueError(f"ERROR: file [{FILENAME}] does not exist at [{FILEPATH}]")
    return X, t_delta

def wrap_loader(loader, filepath):
    """Wrap data loader in datetime delta"""
    a = datetime.now()
    X = loader(filepath)
    b = datetime.now() - a
    return X, b

def pretty_print(txt: str, time_delta):
    print(f"\033[;1m{txt}\033[0;0m\033[1;31m {time_delta}\033[0;0m")

def exec(X: np.ndarray, NUM_CLUSTERS: int):
    print(X.shape, X.dtype)

    # Init model
    m = KMeans(NUM_CLUSTERS, X.shape[-1])
    a = datetime.now()
    # Train model
    m.train(X)
    # Take time delta
    t_delta = datetime.now() - a
    return m, X, t_delta

if __name__ == "__main__":
    main()