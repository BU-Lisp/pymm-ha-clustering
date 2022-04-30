#!/bin/bash

for k in 2 3 4 5 6 7 8 9 10
do
    for p in 1
    do
        echo "$k $p"
        numactl --cpunodebind=1 --membind=1 python3 run_kmeans.py ~/../data/pymm-ha-clustering/ /mnt/nvme1/fault_tolerant/dram/mnist/kmeans_k=$k,p=$p/ /mnt/pmem1/fault_tolerant/kmeans/ mnist.npz $k --save_frequency $p --num_repetitions 3 --shelf_size_mb=50000
    done
done

