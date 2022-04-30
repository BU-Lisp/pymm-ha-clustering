#!/bin/bash

for eps in 0.5 1.0
do
    for npts in 3 6
    do
        for p in 1
        do
            echo "$eps $npts $p"
            numactl --cpunodebind=1 --membind=1 python3 run_dbscan.py ~/../data/pymm-ha-clustering/ /mnt/nvme1/fault_tolerant/dram/musicnet/dbscan_eps=$eps,npts=$npts,p=$p/ /mnt/pmem1/fault_tolerant/dbscan/ point_cloud.npy $eps $npts --save_frequency $p --num_repetitions 3 --shelf_size_mb=50000
        done
    done
done

