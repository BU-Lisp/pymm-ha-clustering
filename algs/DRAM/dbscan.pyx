# SYSTEM IMPORTS
from typing import Callable, List
import numpy as np
cimport numpy as np

np.import_array()

# from libcpp.list cimport list as cpplist


# ctypedef np.int_t np.int
# ctypedef np.float32_t np.float32

# PYTHON PROJECT IMPORTS


def l2_distance(X: np.ndarray, pt: np.ndarray) -> np.ndarray:
    return np.sqrt(((X - pt.reshape(1,-1))**2).sum(axis=-1, keepdims=True))


NO_CLUSTER_CLUSTER_IDX: int = -1


class DBScan(object):
    def __init__(self, epsilon: float, min_points: int = 5,
                 distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> None:
        self.epsilon: float = float(epsilon)
        self.min_points: int = int(min_points)

        self.distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = distance_func
        if self.distance_func is None:
            self.distance_func = l2_distance

        self.assignments: np.ndarray = None
        self.num_clusters: int = 0

    def expand_cluster(self, X, pt_idx, current_cluster_idx) -> bool:
        # print("examining pt [%s]" % pt_idx)

        # ctypedef np.ndarray[int, ndim=1, mode="c"] IDX_ARRAY

        # go point by point and determine if clusters exist
        cdef np.ndarray[np.int_t, ndim=1] neighbor_idxs
        cdef np.ndarray[np.int_t, ndim=1] pt_idxs
        cdef int idx


        neighbor_idxs = (self.distance_func(X, X[pt_idx]).reshape(-1) < self.epsilon).nonzero()[0]

        if neighbor_idxs.shape[0] < self.min_points:
            # this point is either a noise point or a border point
            # if the point is a noise point then we can leave the assignment to its default value (of no cluster)
            # if the point is a border point, it's assignment will be made upon detection of the neighboring core pt
            return False

        # otherwise this point is a core point (it & its neighbors belong to the same cluster)
        self.assignments[neighbor_idxs] = current_cluster_idx

        # now find the rest of the cluster
        # need to use an iterative algorithm (instead of a recursive one) so this will scale
        pts_to_visit: List[np.ndarray] = [neighbor_idxs]

        while len(pts_to_visit) > 0:

            pt_idxs = pts_to_visit.pop(0)

            idx = 0
            for idx in pt_idxs.shape[0]:
                if self.assignments[pt_idxs[idx]] == NO_CLUSTER_CLUSTER_IDX:
                   neighbor_idxs = (self.distance_func(X, X[pt_idxs[idx]]).reshape(-1) < self.epsilon).nonzero()[0]

                if neighbor_idxs.shape[0] >= self.min_points:
                    self.assignments[neighbor_idxs] = current_cluster_idx
                    pts_to_visit.append(neighbor_idxs)
        return True

    def train(self, X: np.ndarray,
              monitor_func: Callable[["DBScan"], None] = None) -> None:
        self.assignments = np.full((X.shape[0], 1), NO_CLUSTER_CLUSTER_IDX, dtype=int)

        cdef int pt_idx
        for pt_idx in range(X.shape[0]):
            if self.assignments[pt_idx] == NO_CLUSTER_CLUSTER_IDX:
                if self.expand_cluster(X, pt_idx, self.num_clusters):
                    self.num_clusters += 1
            if monitor_func is not None:
                monitor_func(self)
    
    def save(self, fp: str) -> None:
        # A method which saves this object to the filepath (fp) using the numpy save_npz method
        model = np.array([self.assignments, self.num_clusters])
        np.savez(fp, model)
    

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
