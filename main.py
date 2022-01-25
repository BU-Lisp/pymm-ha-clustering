import sys, getopt
import os
import argparse

import numpy as np

from algs.kmeans import KMeans

from data.load_mnist import load as load_mnist
from data.load_imagenet2012 import load as load_imagenet
from data.load_musicnet import load as load_musicnet

def main() -> np.ndarray:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    parser.add_argument("--num-clusters", type=int, required=True)
    args = parser.parse_args()

    NUM_CLUSTERS: int = args.num_clusters

    FILEPATH: str = args.dataset_path
    if not os.path.exists(FILEPATH):
        os.makedirs(FILEPATH)
    FILENAME: str = args.dataset_path.rsplit('/', 1)[-1]
    FILEPATH2: str = FILEPATH[:-len(FILENAME)]

    # FILEPATH: str = args.dataset_path.replace(FILENAME, '')
    # if not os.path.exists(FILEPATH):
    #     os.makedirs(FILEPATH)

    if FILENAME == "mnist.npz":
        X = load_mnist(FILEPATH)
    elif FILENAME == "imagenet2012_500k.npy" or "imagenet2012_800k.npy":
        X = load_imagenet(FILEPATH)
    elif FILENAME == "point_cloud.npy":
        X = load_musicnet(FILEPATH)

    print(X.shape, X.dtype)

    m = KMeans(NUM_CLUSTERS, X.shape[-1])
    m.train(X)
    print(X)
    print(m.centers, m.cost(X))

if __name__ == "__main__":
    main()