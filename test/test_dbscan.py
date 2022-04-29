import torch as pt
import clusterutils
import numpy as np
from tqdm import tqdm

X = np.array([[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10], [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]).T
epsilon = 0.5
min_pts = 2
X_pt = pt.from_numpy(X)
assignment_vec = pt.zeros(X_pt.size(0), 1)


with tqdm(total=X_pt.size(0)) as pbar:
    def update():
        pbar.update(1)

    clusterutils.dbscan_train(X_pt, epsilon, min_pts, assignment_vec, update)

