#!/bin/bash

CPU_ID=1
DATASET="mnist"
DATASET_FILE="mnist.npz"

for k in 2 3 4 5 6 7 8 9 10
do
    numactl --cpunodebind=$CPU_ID --membind=$CPU_ID python3 run_gmm.py ~/../data/pymm-ha-clustering/ /mnt/nvme${CPU_ID}/fault_tolerant/dram/$DATASET/gmm_k=${k}_better_recordings/ /mnt/pmem${CPU_ID}/fault_tolerant/gmm/ $DATASET_FILE $k --num_repetitions 3 --batch_size 100 --shelf_size_mb=50000 --gpu_id=$CPU_ID
done

