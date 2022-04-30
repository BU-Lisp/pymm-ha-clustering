# SYSTEM IMPORTS
from typing import Dict, Callable
import torch as pt
import clusterutils
import numpy as np


# PYTHON PROJECT IMPORTS


class DBScan(object):
    def __init__(self, epsilon: float, min_points: int = 5) -> None:
        self.epsilon: float = float(epsilon)
        self.min_points: int = int(min_points)

        self.assignments: np.ndarray = None
        self.num_clusters: int = 0


    def train(self, X: pt.Tensor,
              monitor_func: Callable[[None], None] = None) -> None:
        self.assignment_vec: pt.Tensor = pt.zeros(X.size(0), 1, dtype=int)

        if monitor_func is None:
            monitor_func = lambda: ...

        self.num_clusters = clusterutils.dbscan_train(X, self.epsilon, self.min_points,
                                                      self.assignment_vec, monitor_func)


    def save(self, path: str) -> None:
        np.savez_compressed(path, num_clusters=self.num_clusters, assignments=self.assignment_vec.numpy(),
                            epsilon=self.epsilon, min_points=self.min_points)

    def load(self, path: str) -> "DBScan":
        data_dict: Dict[Union[int, float, np.ndarray]] = np.load(path)

        self.epsilon = data_dict["epsilon"]
        self.min_points = data_dict["min_points"]
        self.num_clusters = data_dict["num_clusters"]
        self.assignment_vec = pt.from_numpy(data_dict["assignments"])

        return self


def main() -> None:
    X = pt.from_numpy(np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]).T)
    epsilon = 0.5
    min_points = 2

    m = DBScan(epsilon, min_points=min_points)
    m.train(X)
    print(X)
    print(m.assignment_vec, m.num_clusters)

if __name__ == "__main__":
    main()

