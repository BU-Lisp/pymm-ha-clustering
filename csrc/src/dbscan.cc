// SYSTEM INCLUDES
#include <iostream>
#include <list>


// C++ PROJECT INCLUDES
#include "dbscan/dbscan.h"


torch::Tensor
euclidean_dist(const torch::Tensor X,
               const torch::Tensor pt)
{
    return torch::sqrt(torch::sum(torch::pow(X - pt.view({1, -1}), 2), {-1}, true));
}


bool
expand_cluster(const torch::Tensor X,
               int64_t pt_idx,
               const int64_t current_cluster_idx,
               const float epsilon,
               const int64_t min_pts,
               torch::Tensor assignment_vec)
{
    torch::Tensor neighbor_idxs = torch::nonzero(euclidean_dist(X, X.index({pt_idx})).view({-1}) < epsilon);

    if(neighbor_idxs.size(0) < min_pts)
    {
        return false;
    }

    assignment_vec.index_put_({neighbor_idxs}, current_cluster_idx);

    std::list<torch::Tensor> pts_to_visit = {neighbor_idxs};
    while(!pts_to_visit.empty())
    {
        torch::Tensor pt_idxs = pts_to_visit.front();
        pts_to_visit.pop_front();

        for(int64_t idx = 0; idx < pt_idxs.size(0); ++idx)
        {
            pt_idx = pt_idxs.index({idx}).item<int64_t>();
            if(assignment_vec.index({pt_idx}).item<int64_t>() == NO_CLUSTER_CLUSTER_IDX)
            {
                neighbor_idxs = torch::nonzero(euclidean_dist(X, X.index({pt_idx})).view({-1}) < epsilon);

                if(neighbor_idxs.size(0) < min_pts)
                {
                    continue;
                }

                assignment_vec.index_put_({neighbor_idxs}, current_cluster_idx);
                pts_to_visit.push_back(neighbor_idxs);
            }
        }
    }

    return true;
}


int64_t
dbscan_train(const torch::Tensor X,
             const float epsilon,
             const int64_t min_pts,
             torch::Tensor assignment_vec,
             std::function<void()>& monitor_func)
{
    int64_t num_clusters = 0;
    assignment_vec.index_put_({torch::indexing::Slice()}, NO_CLUSTER_CLUSTER_IDX);

    for(int64_t pt_idx = 0; pt_idx < X.size(0); ++pt_idx)
    {
        if(assignment_vec.index({pt_idx}).item<int64_t>() == NO_CLUSTER_CLUSTER_IDX)
        {
            if(expand_cluster(X, pt_idx, num_clusters, epsilon, min_pts, assignment_vec))
            {
                num_clusters++;
            }
        }

        monitor_func();
    }

    return num_clusters;
}

