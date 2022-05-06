#!/bin/bash

CPU_ID=1
DATASET="mnist"
DATASET_FILE="mnist.npz"

for k in 2 3 4 5 6 7 8 9 10
do
    echo "$k"
    numactl --cpunodebind=$CPU_ID --membind=$CPU_ID python3 run_kmeans.py ~/../data/pymm-ha-clustering/ /mnt/nvme$CPU_ID/fault_tolerant/dram/$DATASET/kmeans_k=${k}_better_recordings/ /mnt/pmem$CPU_ID/fault_tolerant/kmeans/ $DATASET_FILE $k --num_repetitions 3 --shelf_size_mb=5000
done

