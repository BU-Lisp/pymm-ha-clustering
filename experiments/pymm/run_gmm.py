# SYSTEM IMPORTS
from typing import List
from tqdm import tqdm
import argparse
import os
import numpy as np
import sys
import time
import torch as pt
import pymm


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_,
              os.path.join(_cd_, ".."),
              os.path.join(_cd_, "..", ".."),]:
              #os.path.join(_cd_, "..", "..", "algs", "DRAM")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from algs.PYMM.gmm import GMM
import data.load_mnist as mnist
import data.load_musicnet as musicnet
import data.load_imagenet2012 as imagenet


def main() -> None:
    start_script_time = time.time()

    # common experiment arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to load data from")
    parser.add_argument("out_dir", type=str, help="where to write the results to")
    parser.add_argument("shelf_path", type=str, help="path to shelf")
    parser.add_argument("dataset", type=str, choices=[mnist.FILENAME, musicnet.FILENAME] + imagenet.FILE_CHOICES, help="which dataset to load")

    # data type specific arguments
    parser.add_argument("k", type=float, help="number of clusters (i.e. gaussians)")
    parser.add_argument("--max_iter", type=int, default=int(1e5), help="max number of training iterations")
    parser.add_argument("--batch_size", type=int, default=5000, help="batch size for processing")

    # common default arguments
    parser.add_argument("--num_repetitions", type=int, default=1, help="number of times to repeat the experiment")
    parser.add_argument("--save_frequency", type=int, default=1, help="save model ever XXX epochs")
    parser.add_argument("--shelf_size_mb", type=int, default=50000, help="mb size for shelf")
    parser.add_argument("--gpu_id", type=int, default="cpu", help="what gpu to run on")
    args = parser.parse_args()

    shelf = pymm.shelf("gmm_shelf", size_mb=args.shelf_size_mb, pmem_path=args.shelf_path)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    results_file: str = os.path.join(args.out_dir, "runtimes.npz")
    model_compressed_file: str = os.path.join(args.out_dir, "model_compressed.npz")
    model_file: str = os.path.join(args.out_dir, "mode_uncompressed.npz")

    loading_times: np.ndarray = np.zeros(args.num_repetitions, dtype=float)
    model_creation_times: np.ndarray = np.zeros_like(loading_times)
    epoch_training_times: List[List[float]] = list()
    model_training_times: np.ndarray = np.zeros_like(loading_times)
    script_runtime: float = None

    for experiment_idx in range(args.num_repetitions):
        epoch_training_times.append(list())

        start_loading_time = time.time()
        X = None
        filepath = os.path.join(args.data_dir, args.dataset)
        if "mnist" in args.dataset:
            X = mnist.load(filepath)
        elif "imagenet" in args.dataset:
            X = imagenet.load(filepath)
        else:
            X = musicnet.load(filepath)
        X = pt.from_numpy(X).to(pt.float32)# .to(0)
        loading_times[experiment_idx] = time.time() - start_loading_time

        start_model_creation_time = time.time()
        m = GMM(X.shape[-1], args.k, shelf).to(args.gpu_id)
        model_creation_times[experiment_idx] = time.time() - start_model_creation_time

        with tqdm(total=args.max_iter, desc="training model @ experiment %s" % experiment_idx) as pbar:
            start_training_time = time.time()
            epoch_training_times[-1].append(start_training_time)

            def record_times_and_progress_bar_update(model) -> None:
                end_epoch_time = time.time()

                epoch_training_times[-1][-1] = end_epoch_time - epoch_training_times[-1][-1]
                pbar.update(1)

                epoch_training_times[-1].append(time.time())

            m.train(X, monitor_func=record_times_and_progress_bar_update, max_iter=args.max_iter,
                    batch_size=args.batch_size)
            model_training_times[experiment_idx] = time.time() - start_training_time
    script_runtime = time.time() - start_script_time

    np.savez_compressed(results_file, loading_times=loading_times,
                                      model_creation_times=model_creation_times,
                                      epoch_training_times=epoch_training_times,
                                      model_training_times=model_training_times,
                                      script_runtime=script_runtime)

if __name__ == "__main__":
    main()


