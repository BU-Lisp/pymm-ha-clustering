# SYSTEM IMPORTS
from typing import Callable, List
import numpy as np

import pymm

np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS




def l2_squared_distance(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((X - center.reshape(1,-1))**2).sum(axis=-1, keepdims=True)


class KMeans(object):

    def __init__(self, shelf: pymm.shelf, distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> None:

        self.s = shelf

        # NOTE: When trying not to copy k/num_features to DRAM in __init__ use of self.s.k in line 65 produces:
        # TypeError: 'shelved_float_number' object cannot be interpreted as an integer
        self.k: int = int(shelf.k)
        self.num_features: int = int(shelf.num_features)

        self.distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = distance_func
        if self.distance_func is None:
            self.distance_func = l2_squared_distance

    def init_centers(self, X: np.ndarray) -> None:
        self.s.centers = np.random.randn(int(self.k), int(self.num_features))
        # print(self.centers.shape)

        # kmeans++ initialization
        not_chosen_mask: np.ndarray = np.ones(X.shape[0], dtype=bool)
        pt_idxs: np.ndarray = np.arange(X.shape[0])

        # step 1) first choose one center uniformly at random among the data points
        center_idx: int = np.random.randint(0, X.shape[0])
        self.s.centers[0] = X[center_idx]
        not_chosen_mask[center_idx] = False

        num_centers_processed: int = 1
        while num_centers_processed < self.k:
            X_view: np.ndarray = X[not_chosen_mask]
            # step 2) for each data point x (not chosen), compute distance between x and it's closest center
            min_dists_squared: np.ndarray = np.hstack([self.distance_func(X_view, self.s.centers[c_idx]).reshape(-1, 1)
                                                       for c_idx in range(num_centers_processed)]).min(axis=1) ** 2

            # step 3) choose new data point at random using probs proportional to D(x)^2
            weights: np.ndarray = min_dists_squared / min_dists_squared.sum()
            center_idx = np.random.choice(X_view.shape[0], p=weights)
            self.s.centers[num_centers_processed] = X_view[center_idx]
            not_chosen_mask[pt_idxs[not_chosen_mask][center_idx]] = False
            num_centers_processed += 1

    def assign(self, X: np.ndarray) -> np.ndarray:
        X_distances: np.ndarray = np.hstack([self.distance_func(X, self.s.centers[c_idx]).reshape(-1,1)
                                             for c_idx in range(self.k)])
        return np.argmin(X_distances, axis=-1)

    def cost(self, X: np.ndarray) -> float:
        cost: float = 0

        X_assignments: np.ndarray = self.assign(X)
        for cluster_idx in range(self.k):
            X_assigned_mask: np.ndarray = X_assignments == cluster_idx
            num_assigned: int = X_assigned_mask.sum()
            if num_assigned > 0:
                cost += self.distance_func(X[X_assigned_mask], self.s.centers[cluster_idx]).sum()

        return cost

    def train_iter(self, X: np.ndarray) -> None:
        # assign points to clusters (hard counts)
        X_assignments: np.ndarray = self.assign(X)
        # print(X_assignments)

        # recompute clusters from assignments
        for cluster_idx in range(self.k):
            X_assigned_mask: np.ndarray = (X_assignments.reshape(-1) == cluster_idx)
            # print(X_assigned_mask)
            num_assigned: int = X_assigned_mask.sum()
            if num_assigned > 0:
                self.s.centers[cluster_idx] = X[X_assigned_mask].mean(axis=0)

    def train(self, X: np.ndarray, epsilon: float = 1e-9, max_iter: int = 1e6,
              monitor_func: Callable[["KMeans"], None] = None) -> None:
        # structure of iterative algorithm: while (dont give up) and (havent converged): do stuff
        current_iter: int = 0
        prev_cost: float = np.inf
        current_cost: float = 0.0


        if self.s.centers is None:
            self.init_centers(X)
        while current_iter < max_iter and abs(prev_cost - current_cost) > epsilon:
            self.train_iter(X)

            prev_cost = current_cost
            current_cost = self.cost(X)
            current_iter += 1

            if monitor_func is not None:
                monitor_func(self)


def main() -> None:

    s = pymm.shelf('kmeans_test', pmem_path='/mnt/pmem0', size_mb=1024)

    X = np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]).T
    k: int = 3

    m = KMeans(k, X.shape[-1])
    m.train(X)
    print(X)
    print(m.centers, m.cost(X))


if __name__ == "__main__":
    main()
