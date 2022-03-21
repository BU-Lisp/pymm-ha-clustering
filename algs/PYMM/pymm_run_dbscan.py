# SYSTEM IMPORTS
from tqdm import tqdm
import argparse
import time
import os
import sys
import pymm


# make sure we can import the files we need from this repo
# we can do this by adding the appropriate paths to sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from pymm_dbscan import DBScan
import pymm_load_mnist as mnist
import pymm_load_musicnet as musicnet
import pymm_load_imagenet2012 as imagenet


def main() -> None:
    start_script_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to load data from")
    parser.add_argument("epsilon", type=float, help="DBSCAN Parameter epsilon")
    parser.add_argument("min_points", type=int, help="DBSCAN Parameter min_points")
    parser.add_argument("dataset", type=str, choices=[mnist.FILENAME, musicnet.FILENAME] + imagenet.FILE_CHOICES, help="which dataset to load")
    parser.add_argument("shelf_size", type=int, help="PYMM shelf size (mb)")
    args = parser.parse_args()

    ds = pymm.shelf('dataset_shelf',size_mb=args.shelf_size,pmem_path='/mnt/pmem0/will_pymm_dbscan',force_new=True)     #initialize a pymm shelf for training dataset

    start_loading_time = time.time()
    ds.X = None
    filepath = os.path.join(args.data_dir, args.dataset)
    if "mnist" in args.dataset:
        ds.X = mnist.load(filepath)
    elif "imagenet" in args.dataset:
        ds.X = imagenet.load(filepath)
    else:
        ds.X = musicnet.load(filepath)
    load_runtime = time.time() - start_loading_time

    start_model_creation_time = time.time()
    m = DBScan(epsilon=args.epsilon, min_points=args.min_points)
    model_creation_runtime = time.time() - start_model_creation_time

    with tqdm(total=ds.X.shape[0], desc="training model") as pbar:
        def progress_bar_update(model: DBScan) -> None:
            pbar.update(1)

        start_training_time = time.time()
        m.train(ds.X, monitor_func=progress_bar_update)
        training_runtime = time.time() - start_training_time
        script_runtime = time.time() - start_script_time

    print("data loading time %s" % load_runtime)
    print("model creation time %s" % model_creation_runtime)
    print("training time %s" % training_runtime)
    print("total script time %s" % script_runtime)


if __name__ == "__main__":
    main()
