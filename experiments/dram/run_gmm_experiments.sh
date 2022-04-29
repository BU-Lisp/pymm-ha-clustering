#!/bin/bash

for k in 2 3 4 5 6 7 8 9 10
do
    for p in 1
    do
        numactl --cpunodebind=0 --membind=0 python3 run_gmm.py ~/../data/pymm-ha-clustering/ /mnt/nvme0/fault_tolerant/dram/musicnet/gmm_k=$k,p=$p/ /mnt/pmem0/fault_tolerant/gmm/ point_cloud.npy $k --save_frequency $p --num_repetitions 3 --batch_size 10000 --shelf_size_mb=50000
    done
done

