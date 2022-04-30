#!/bin/bash

for k in 2 3 4 5 6 7 8 9 10
do
    echo "$k"
    numactl --cpunodebind=1 --membind=1 python3 run_kmeans.py ~/../data/pymm-ha-clustering/ /mnt/nvme1/fault_tolerant/pymm/mnist/kmeans_k=$k/ /mnt/pmem1/fault_tolerant/kmeans/ mnist.npz $k --num_repetitions 3 --shelf_size_mb=5000 --shelf_name=kmeans_mnist_pymm_shelf
done

