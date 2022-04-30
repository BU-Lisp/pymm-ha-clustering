#pragma once
#ifndef _FAULT_TOLERANT_DBSCAN_H_
#define _FAULT_TOLERANT_DBSCAN_H_


// SYSTEM INCLUDES
#include <functional>
#include <torch/extension.h>


// C++ PROJECT INCLUDES


const int64_t NO_CLUSTER_CLUSTER_IDX = -1;


/**
    Utility functions for dbscan with PyTorch tensors
 */
torch::Tensor
euclidean_dist(const torch::Tensor X,
               const torch::Tensor pt);


bool
expand_cluster(const torch::Tensor X,
               int64_t pt_idx,
               const int64_t current_cluster_idx,
               const float epsilon,
               const int64_t min_pts,
               torch::Tensor assignment_vec);


int64_t
dbscan_train(const torch::Tensor X,
             const float epsilon,
             const int64_t min_pts,
             torch::Tensor assignment_vec,
             std::function<void()>& monitor_func);




#endif // end of _FAULT_TOLERANT_DBSCAN_H_

