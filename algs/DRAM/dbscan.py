# SYSTEM IMPORTS
from typing import Callable, List
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

    def expand_cluster(self, X: np.ndarray, current_pt_idx: int, current_cluster_idx: int,
                       unprocessed_pt_mask: np.ndarray, all_idxs: np.ndarray) -> int:
        # print("examining pt [%s]" % pt_idx)
        cluster_size: int = 0
        unprocessed_pt_mask[current_pt_idx] = False

        X_cluster_to_expand: np.ndarray = X[current_pt_idx, :].reshape(1,-1)
        X_remaining = X[unprocessed_pt_mask, :]

        # go point by point and determine if clusters exist
        neighbor_idxs: np.ndarray = all_idxs[unprocessed_pt_mask][cdist(X_remaining, X_cluster_to_expand).reshape(-1) < self.epsilon]

        if neighbor_idxs.shape[0] >= self.min_points:
            cluster_size += (1 + neighbor_idxs.shape[0])

            # mark all the neighbors as belonging to the cluster
            unprocessed_pt_mask[neighbor_idxs] = False
            self.assignments[current_pt_idx] = current_cluster_idx
            self.assignments[neighbor_idxs] = current_cluster_idx

            # now find the rest of the cluster
            # need to use an iterative algorithm (instead of a recursive one) so this will scale
            pts_to_expand: List[np.ndarray] = [neighbor_idxs]
            while len(pts_to_expand) > 0:
                pt_idxs: np.ndarray = pts_to_expand.pop(0)

                # slice out
                X_cluster_to_expand = X[pt_idxs, :]
                X_remaining = X[unprocessed_pt_mask, :] # reduce data size

                neighbor_masks: np.ndarray = cdist(X_remaining, X[pt_idxs]) < self.epsilon
                # print(neighbor_masks)
                interior_pts_mask: np.ndarray = np.sum(neighbor_masks, axis=0, dtype=int) >= self.min_points
                interior_pts_neighbors_mask: np.ndarray = np.sum(neighbor_masks[:, interior_pts_mask],
                                                                 axis=-1, dtype=bool)

                # print(all_idxs[unprocessed_pt_mask].shape, interior_pts_neighbors_mask.shape)
                neighbor_idxs = all_idxs[unprocessed_pt_mask][interior_pts_neighbors_mask]

                if neighbor_idxs.shape[0] > 0:
                    unprocessed_pt_mask[neighbor_idxs] = False
                    self.assignments[neighbor_idxs] = current_cluster_idx
                    cluster_size += neighbor_idxs.shape[0]

                    pts_to_expand.append(neighbor_idxs)

        return cluster_size

    def train(self, X: np.ndarray,
              monitor_func: Callable[["DBScan"], None] = None) -> None:
        self.assignments = np.full((X.shape[0], 1), NO_CLUSTER_CLUSTER_IDX, dtype=int)

        unprocessed_pt_mask: np.ndarray = np.ones(X.shape[0], dtype=bool)
        all_idxs: np.ndarray = np.arange(X.shape[0])

        self.num_pts_processed = 0
        num_pts: int = X.shape[0]
        current_pt_idx: int = 0
        while self.num_pts_processed < num_pts:
            # print(num_pts_processed)

            if self.assignments[current_pt_idx] == NO_CLUSTER_CLUSTER_IDX:
                cluster_size: int = self.expand_cluster(X, current_pt_idx, self.num_clusters,
                                                        unprocessed_pt_mask, all_idxs)
                self.num_pts_processed += max(1, cluster_size)

                if cluster_size > 0:
                    self.num_clusters += 1
            current_pt_idx += 1

            if monitor_func is not None:
                monitor_func(self)
    
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
