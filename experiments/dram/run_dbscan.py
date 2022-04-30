# SYSTEM IMPORTS
from typing import List
from tqdm import tqdm
import argparse
import os
import numpy as np
import sys
import time
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
from algs.DRAM.dbscan import DBScan
# from dbscan import DBScan
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
    parser.add_argument("epsilon", type=float, help="DBSCAN Parameter epsilon")
    parser.add_argument("min_points", type=int, help="DBSCAN Parameter min_points")
    parser.add_argument("--max_iter", type=int, default=int(1e5), help="max number of training iterations")

    # common default arguments
    parser.add_argument("--num_repetitions", type=int, default=1, help="number of times to repeat the experiment")
    parser.add_argument("--save_frequency", type=int, default=1, help="save model ever XXX epochs")
    parser.add_argument("--shelf_size_mb", type=int, default=50000, help="mb size for shelf")
    args = parser.parse_args()

    shelf = pymm.shelf("dbscan_shelf", size_mb=args.shelf_size_mb, pmem_path=args.shelf_path)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    results_file: str = os.path.join(args.out_dir, "runtimes.npz")
    model_compressed_file: str = os.path.join(args.out_dir, "model_compressed.npz")
    model_file: str = os.path.join(args.out_dir, "mode_uncompressed.npz")

    loading_times: np.ndarray = np.zeros(args.num_repetitions, dtype=float)
    model_creation_times: np.ndarray = np.zeros_like(loading_times)
    model_nvme_compressed_saving_times: List[List[float]] = list() # dont know how many epochs each training call may take
    model_nvme_saving_times: List[List[float]] = list()
    model_pymm_saving_times: List[List[float]] = list()
    epoch_training_times: List[List[float]] = list()
    model_training_times: np.ndarray = np.zeros_like(loading_times)
    script_runtime: float = None

    for experiment_idx in range(args.num_repetitions):
        model_nvme_compressed_saving_times.append(list())
        model_nvme_saving_times.append(list())
        model_pymm_saving_times.append(list())
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
        loading_times[experiment_idx] = time.time() - start_loading_time

        start_model_creation_time = time.time()
        m = DBScan(epsilon=args.epsilon, min_points=args.min_points)
        model_creation_times[experiment_idx] = time.time() - start_model_creation_time

        with tqdm(total=args.max_iter, desc="training model @ experiment %s" % experiment_idx) as pbar:
            start_training_time = time.time()
            epoch_training_times[-1].append(start_training_time)

            def record_times_and_progress_bar_update(model) -> None:

                """
                end_epoch_time = time.time()

                num_epochs = len(epoch_training_times[-1])
                if num_epochs % args.save_frequency == 0:
                    start_saving_time = time.time()
                    model.save_compressed(model_compressed_file)
                    model_nvme_compressed_saving_times[-1].append(time.time() - start_saving_time)

                    start_saving_time = time.time()
                    model.save(model_file)
                    model_nvme_saving_times[-1].append(time.time() - start_saving_time)

                    start_saving_time = time.time()
                    model.save_shelf(shelf)
                    model_pymm_saving_times[-1].append(time.time() - start_saving_time)

                epoch_training_times[-1][-1] = end_epoch_time - epoch_training_times[-1][-1]
                """
                pbar.update(model.num_pts_processed - pbar.n)

                epoch_training_times[-1].append(time.time())

            m.train(X, monitor_func=record_times_and_progress_bar_update)
            model_training_times[experiment_idx] = time.time() - start_training_time
    script_runtime = time.time() - start_script_time

    np.savez_compressed(results_file, loading_times=loading_times,
                                      model_creation_times=model_creation_times,
                                      model_nvme_compressed_saving_times=model_nvme_compressed_saving_times,
                                      model_nvme_saving_times=model_nvme_saving_times,
                                      model_pymm_saving_times=model_pymm_saving_times,
                                      epoch_training_times=epoch_training_times,
                                      model_training_times=model_training_times,
                                      script_runtime=script_runtime)

if __name__ == "__main__":
    main()


