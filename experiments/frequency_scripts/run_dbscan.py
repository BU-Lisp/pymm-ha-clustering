# SYSTEM IMPORTS
from tqdm import tqdm
import argparse
import time
import os
import sys


# make sure we can import the files we need from this repo
# we can do this by adding the appropriate paths to sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from algs.dbscan import DBScan
import data.load_mnist as mnist
import data.load_musicnet as musicnet
import data.load_imagenet2012 as imagenet


def main() -> None:
    start_script_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to load data from")
    parser.add_argument("epsilon", type=float, help="DBSCAN Parameter epsilon")
    parser.add_argument("min_points", type=int, help="DBSCAN Parameter min_points")
    parser.add_argument("dataset", type=str, choices=[mnist.FILENAME, musicnet.FILENAME] + imagenet.FILE_CHOICES, help="which dataset to load")
    args = parser.parse_args()

    start_loading_time = time.time()
    X = None
    filepath = os.path.join(args.data_dir, args.dataset)
    if "mnist" in args.dataset:
        X = mnist.load(filepath)
    elif "imagenet" in args.dataset:
        X = imagenet.load(filepath)
    else:
        X = musicnet.load(filepath)
    load_runtime = time.time() - start_loading_time

    start_model_creation_time = time.time()
    m = DBScan(epsilon=args.epsilon, min_points=args.min_points)
    model_creation_runtime = time.time() - start_model_creation_time

    with tqdm(total=X.shape[0], desc="training model") as pbar:
        def progress_bar_update(model: DBScan) -> None:
            pbar.update(1)

        start_training_time = time.time()
        m.train(X, monitor_func=progress_bar_update)
        training_runtime = time.time() - start_training_time
        script_runtime = time.time() - start_script_time

    print("data loading time %s" % load_runtime)
    print("model creation time %s" % model_creation_runtime)
    print("training time %s" % training_runtime)
    print("total script time %s" % script_runtime)


if __name__ == "__main__":
    main()
