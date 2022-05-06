# SYSTEM IMPORTS
from typing import Callable, List
import numpy as np
import sys
import pymm


np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS




def l2_squared_distance_old(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((X - center.reshape(1,-1))**2).sum(axis=-1, keepdims=True)

def l2_squared_distance(X: np.ndarray, center: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    return ((X - center.reshape(1,-1))**2).sum(axis=-1, out=out)


class KMeans(object):
    def __init__(self, k: int, num_features: int, shelf,
                 distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> None:
        self.shelf = shelf
        self.shelf.k: int = int(k)
        self.shelf.num_features: int = int(num_features)
        # print(k, num_features)
        self.shelf.centers: pymm.ndarray = np.zeros((self.shelf.k, self.shelf.num_features,), dtype=float)

        # buffers for intermediary ops
        self.shelf.distance_buffer: pymm.ndarray = None
        self.shelf.assignments: pymm.ndarray = None

        self.distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = distance_func
        if self.distance_func is None:
            self.distance_func = l2_squared_distance

    def init_centers(self, X: np.ndarray) -> None:
        # self.shelf.centers = np.random.randn(self.shelf.k, self.shelf.num_features)
        np.copyto(self.shelf.centers, np.random.randn(self.shelf.k, self.shelf.num_features))
        # print(self.centers.shape)

        # kmeans++ initialization
        not_chosen_mask: np.ndarray = np.ones(X.shape[0], dtype=bool)
        pt_idxs: np.ndarray = np.arange(X.shape[0])

        # step 1) first choose one center uniformly at random among the data points
        center_idx: int = np.random.randint(0, X.shape[0])

        # self.shelf.centers[0] = X[center_idx]
        np.copyto(self.shelf.centers[0,:], X[center_idx,:])

        not_chosen_mask[center_idx] = False

        num_centers_processed: int = 1
        while num_centers_processed < self.shelf.k:
            X_view: np.ndarray = X[not_chosen_mask]
            # step 2) for each data point x (not chosen), compute distance between x and it's closest center
            min_dists_squared: np.ndarray = np.hstack([self.distance_func(X_view, self.shelf.centers[c_idx,:]).reshape(-1,1)
                                                       for c_idx in range(num_centers_processed)]).min(axis=1) ** 2

            # step 3) choose new data point at random using probs proportional to D(x)^2
            weights: np.ndarray = min_dists_squared / min_dists_squared.sum()
            center_idx = np.random.choice(X_view.shape[0], p=weights)

            # self.shelf.centers[num_centers_processed] = X_view[center_idx]
            np.copyto(self.shelf.centers[num_centers_processed, :], X_view[center_idx,:]) # faster

            not_chosen_mask[pt_idxs[not_chosen_mask][center_idx]] = False
            num_centers_processed += 1

    def assign(self, X: np.ndarray) -> pymm.ndarray:
        # self.shelf.distance_buffer: pymm.ndarray = np.zeros((X.shape[0], 2), dtype=float) # defaults to zeros
        # self.shelf.assignments: pymm.ndarray = pymm.ndarray((X.shape[0],), dtype=int) # defaults to zeros

        # only need to allocate buffer if it doesnt exist or is the wrong shape
        if self.shelf.distance_buffer is None or self.shelf.distance_buffer.shape != (X.shape[0], 2):
            self.shelf.distance_buffer: pymm.ndarray = np.zeros((X.shape[0], 2), dtype=float) # defaults to zeros
        else:
            self.shelf.distance_buffer.fill(0)

        if self.shelf.assignments is None or self.shelf.assignments.shape != (X.shape[0],):
            self.shelf.assignments: pymm.ndarray = np.zeros((X.shape[0],), dtype=int) # defaults to zeros
        else:
            self.shelf.assignments.fill(0)

        arange: np.ndarray = np.arange(X.shape[0])

        self.distance_func(X, self.shelf.centers[0, :], out=self.shelf.distance_buffer[:, 0])

        mins: np.ndarray = np.empty_like(self.shelf.assignments)
        for cluster_idx in range(1, self.shelf.k):
            self.distance_func(X, self.shelf.centers[cluster_idx, :], out=self.shelf.distance_buffer[:, 1])
            np.argmin(self.shelf.distance_buffer, axis=-1, out=mins)

            # if any of mins are 1, then the new value is smaller than the old value
            # min_mask: np.ndarray = mins == 1
            # self.shelf.assignments[mins == 1] = cluster_idx
            # self.shelf.distance_buffer[:, 0] = self.shelf.distance_buffer[arange, mins]

            # np.copyto(dst, src, where=True)
            np.copyto(self.shelf.assignments, np.array([cluster_idx], dtype=self.shelf.assignments.dtype),
                      where=mins==1)
            np.copyto(self.shelf.distance_buffer[:, 0], self.shelf.distance_buffer[arange, mins])
            self.shelf.distance_buffer.persist()

        self.shelf.assignments.persist()
        return self.shelf.assignments

    def cost(self, X: np.ndarray) -> float:
        cost: float = 0

        self.assign(X)
        for cluster_idx in range(self.shelf.k):
            X_assigned_mask: np.ndarray = self.shelf.assignments == cluster_idx
            num_assigned: int = X_assigned_mask.sum()
            if num_assigned > 0:
                cost += self.distance_func(X[X_assigned_mask], self.shelf.centers[cluster_idx]).sum()

        return cost

    def train_iter(self, X: np.ndarray) -> None:
        # assign points to clusters (hard counts)
        self.assign(X)
        # print(X_assignments)

        # recompute clusters from assignments
        for cluster_idx in range(self.shelf.k):
            X_assigned_mask: np.ndarray = (self.shelf.assignments == cluster_idx)
            # print(X_assigned_mask)
            num_assigned: int = X_assigned_mask.sum()
            if num_assigned > 0:
                X[X_assigned_mask].mean(axis=0, out=self.shelf.centers[cluster_idx])

        self.shelf.centers.persist()

    def train(self, X: np.ndarray, epsilon: float = 1e-9, max_iter: int = 1e6, 
                monitor_func: Callable[["KMeans"], None] = None) -> None:
        # structure of iterative algorithm: while (dont give up) and (havent converged): do stuff
        current_iter: int = 0
        prev_cost: float = sys.float_info.max
        current_cost: float = sys.float_info.epsilon


        if self.shelf.centers is None:
            self.init_centers(X)

        # init buffers
        self.shelf.distance_buffer: pymm.ndarray = pymm.ndarray((X.shape[0], 2), dtype=float) # defaults to zeros
        self.shelf.assignments: pymm.ndarray = pymm.ndarray((X.shape[0],), dtype=int) # defaults to zeros

        while current_iter < max_iter and abs(prev_cost - current_cost)/abs(prev_cost) > epsilon:
            self.train_iter(X)

            prev_cost = current_cost
            current_cost = self.cost(X)
            current_iter += 1

            if monitor_func is not None:
                monitor_func(self)

def main() -> None:
    X = np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]).T
    k: int = 3

    shelf = pymm.shelf("kmeans_test_shelf", size_mb=200, pmem_path="/tmp")

    m = KMeans(k, X.shape[-1], shelf)
    m.train(X)
    print(X)
    print(m.shelf.centers, m.cost(X))


if __name__ == "__main__":
    main()
