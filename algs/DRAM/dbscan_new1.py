# SYSTEM IMPORTS
from typing import Callable, List, Tuple
from enum import Enum
from scipy.spatial.distance import cdist
import numpy as np



np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS


NO_CLUSTER_CLUSTER_IDX: int = -1


class DBScan(object):
    def __init__(self, epsilon: float, min_points: int = 5) -> None:
        self.epsilon: float = float(epsilon)
        self.min_points: int = int(min_points)

        self.assignments: np.ndarray = None
        self.num_clusters: int = 0
        self.num_pts_processed: int = 0

    def expand_cluster(self, X_remaining: np.ndarray, current_pt_idx: int, current_cluster_idx: int,
                       remaining_idxs: np.ndarray, unprocessed_pt_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        cluster_size: int = 0

        # print(X_remaining)
        # print("examining pt [%s]" % current_pt_idx)

        # print("examining pt [%s]" % pt_idx)
        local_pt_idxs: np.ndarray = np.arange(X_remaining.shape[0])

        # go point by point and determine if clusters exist
        local_neighbor_mask: np.ndarray = cdist(X_remaining, X_remaining[current_pt_idx, :].reshape(1,-1))\
            .reshape(-1) < self.epsilon
        local_neighbor_idxs: np.ndarray = local_pt_idxs[local_neighbor_mask]
        unprocessed_pt_mask[remaining_idxs[current_pt_idx]] = False

        pts_remaining_mask: np.ndarray = np.ones_like(local_pt_idxs, dtype=bool)
        pts_remaining_mask[current_pt_idx] = False

        # print(local_neighbor_idxs)

        if local_neighbor_idxs.shape[0] >= (self.min_points + 1):
            cluster_size += local_neighbor_idxs.shape[0]

            pts_remaining_mask: np.ndarray = np.ones_like(local_pt_idxs, dtype=bool)
            pts_remaining_mask[local_neighbor_idxs] = False

            # print(pts_remaining_mask)

            self.assignments[remaining_idxs[local_neighbor_idxs]] = current_cluster_idx
            # print(self.assignments)
            unprocessed_pt_mask[remaining_idxs[local_neighbor_idxs]] = False
            remaining_idxs = remaining_idxs[pts_remaining_mask]

            all_pts_in_cluster: List[np.ndarray] = [local_neighbor_idxs]

            pts_to_expand: List[np.ndarray] = [local_neighbor_idxs]
            while len(pts_to_expand) > 0:
                local_neighbor_idxs = pts_to_expand.pop()

                X_to_expand = X_remaining[local_neighbor_idxs, :]
                X_remaining = X_remaining[pts_remaining_mask, :]
                local_pt_idxs = local_pt_idxs[pts_remaining_mask]

                # print(pts_remaining_mask)

                dists_mask: np.ndarray = cdist(X_remaining, X_to_expand) < self.epsilon
                pts_to_expand_mask: np.ndarray = np.sum(dists_mask, axis=0, dtype=int) >= self.min_points

                if np.any(pts_to_expand_mask):
                    local_neighbor_mask = np.sum(dists_mask[:, pts_to_expand_mask], axis=-1, dtype=bool)
                    local_neighbor_idxs = local_pt_idxs[local_neighbor_mask]

                    pts_remaining_mask = pts_remaining_mask[pts_remaining_mask]
                    pts_remaining_mask[local_neighbor_idxs] = False

                    self.assignments[remaining_idxs[local_neighbor_idxs]] = current_cluster_idx
                    cluster_size += local_neighbor_idxs.shape[0]
                    unprocessed_pt_mask[remaining_idxs[local_neighbor_idxs]] = False
                    remaining_idxs = remaining_idxs[pts_remaining_mask]

                    pts_to_expand.append(local_neighbor_idxs)

                # for pt_idx in pt_idxs:
                #     if self.assignments[pt_idx] == NO_CLUSTER_CLUSTER_IDX:
                #         # unassignment pt, examine it
                #         neighbor_idxs = (self.distance_func(X, X[pt_idx]).reshape(-1) < self.epsilon).nonzero()[0]

                #         if neighbor_idxs.shape[0] < self.min_points:
                #             # border point or noise point...do nothing
                #             continue

                #         self.assignments[neighbor_idxs] = current_cluster_idx
                #         pts_to_visit.append(neighbor_idxs)
        else:
            X_remaining = X_remaining[pts_remaining_mask, :]

        # print("assignments", self.assignments)
        # print("cluster size", cluster_size)
        return cluster_size, X_remaining

    def train(self, X: np.ndarray,
              monitor_func: Callable[["DBScan"], None] = None) -> None:
        self.assignments = np.full((X.shape[0], 1), NO_CLUSTER_CLUSTER_IDX, dtype=int)

        unprocessed_pt_mask: np.ndarray = np.ones(X.shape[0], dtype=bool)
        all_idxs: np.ndarray = np.arange(X.shape[0])
        remaining_idxs = all_idxs

        self.num_pts_processed = 0
        num_pts: int = X.shape[0]
        current_pt_idx: int = 0

        """
        print("CALL 1")
        cluster_size, X = self.expand_cluster(X, current_pt_idx, self.num_clusters, all_idxs, unprocessed_pt_mask)
        num_pts_processed += min(1, cluster_size)
        if cluster_size > 0:
            self.num_clusters += 1

        print(cluster_size)
        print(unprocessed_pt_mask)
        print(X.shape, X)
        print()
        print()
        print()

        print("CALL 2")
        remaining_idxs = all_idxs[unprocessed_pt_mask]
        cluster_size, X = self.expand_cluster(X, current_pt_idx, self.num_clusters, remaining_idxs, unprocessed_pt_mask)
        num_pts_processed += min(1, cluster_size)

        print(cluster_size)
        print(unprocessed_pt_mask)
        print(X.shape, X)
        """

        
        while self.num_pts_processed < num_pts:
            # print("loop iter", current_pt_idx, num_pts_processed, self.assignments, X)
            if self.assignments[remaining_idxs[current_pt_idx]] == NO_CLUSTER_CLUSTER_IDX:
                cluster_size, X = self.expand_cluster(X, current_pt_idx, self.num_clusters,
                                                      remaining_idxs, unprocessed_pt_mask)
                self.num_pts_processed += max(1, cluster_size)

                if cluster_size > 0:
                    self.num_clusters += 1

                current_pt_idx = -1 # reset current_pt_idx (will increment to 0 in subsequent line)
            current_pt_idx += 1

            # always want to work with a smaller dataset
            remaining_idxs = all_idxs[unprocessed_pt_mask]

            if monitor_func is not None:
                monitor_func(self)
            # print("end of loop iter")
        

    def save_compressed(self, filepath: str) -> None:
        np.savez_compressed(filepath, epsilon=self.epsilon,
                            min_points=self.min_points,
                            assignments=self.assignments,
                            num_clusters=self.num_clusters)

    def save(self, filepath: str) -> None:
        np.savez(filepath, epsilon=self.epsilon,
                           min_points=self.min_points,
                           assignments=self.assignments,
                           num_clusters=self.num_clusters)

    def save_shelf(self, shelf) -> None:
        shelf.epsilon = self.epsilon
        shelf.min_points = self.min_points
        shelf.assignments = self.assignments
        shelf.num_clusters = self.num_clusters
    

    def load(self, fp: str) -> "DBScan":
        # A method which loads the data contained in filepath (fp) to this object. This method should perform the opposite of the save method
        model = np.load(fp)
        self.assignments = model[0]
        self.num_clusters = model[1]

def main() -> None:
    X = np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]).T
    epsilon = 0.5
    min_points = 2

    m = DBScan(epsilon, min_points=min_points)
    m.train(X)
    print(X)
    print(m.assignments, m.num_clusters)


if __name__ == "__main__":
    main()
