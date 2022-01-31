# SYSTEM IMPORTS
from typing import Callable, List
import numpy as np
from scipy.stats import multivariate_normal, random_correlation
from GMM import GMM 
np.random.seed(12345)




# PYTHON PROJECT IMPORTS


# CONSTANTS
EPSILON = 1e-12

def test_model_1d() -> None:
    print("running 1d test")
    num_samples: int = 100
    num_gaussians: int = 3

    real_mus: np.ndarray = np.array([-4, 4, 0], dtype=float)
    real_vars: np.ndarray = np.array([1.2, 0.8, 2], dtype=float)

    X: np.ndarray = np.vstack([np.random.normal(loc=rmu, scale=rvar, size=num_samples).reshape(-1,1)
                               for rmu, rvar in zip(real_mus, real_vars)])

    lls: List[float] = list()
    def collect_ll_per_iter(m: GMM) -> None:
        lls.append(m.log_likelihood(X))

    # if correctly implemented, the log-likelihood should monotonically increase (or plateau)
    m = GMM(1, num_gaussians)
    print("init ll: %s" % m.log_likelihood(X))
    m.train(X, max_iter=100, monitor_func=collect_ll_per_iter)

    if len(lls) == 0:
        raise RuntimeError("1d test FAILED. No log-likelihoods were recorded")

    # convert lls into np array
    lls = np.array(lls, dtype=float)
    if np.any(lls[:-1] > lls[1:]):
        raise RuntimeError("1d test FAILED. Log-likelihood did not monotonically increase")
    else:
        print("1d test PASSED")


def test_model_2d() -> None:
    print("running 2d test")
    num_samples: int = 100
    num_gaussians: int = 3
    num_features: int = 2

    real_mus: np.ndarray = np.array([[-4,0],
                                     [-2,2],
                                     [1,5]], dtype=float)
    real_covs: np.ndarray = np.stack([random_correlation.rvs([0.4, 1.6]),
                                      random_correlation.rvs([1, 1]),
                                      random_correlation.rvs([1.3, 0.7])], axis=0)

    X: np.ndarray = np.vstack([np.random.multivariate_normal(rmu, rcov, size=num_samples)
                              for rmu, rcov in zip(real_mus, real_covs)]) # Created data with three different clusters
    print("X shape = ", X.shape )
    print(X[:20])
    lls: List[float] = list()
    def collect_ll_per_iter(m: GMM) -> None:
        lls.append(m.log_likelihood(X))

    # if correctly implemented, the log-likelihood should monotonically increase (or plateau)
    m = GMM(num_features, num_gaussians)
    print("init ll: %s" % m.log_likelihood(X))
    m.train(X, max_iter=100, monitor_func=collect_ll_per_iter)

    if len(lls) == 0:
        raise RuntimeError("1d test FAILED. No log-likelihoods were recorded")


    # convert lls into np array
    lls = np.array(lls, dtype=float)
    if np.any(lls[:-1] > lls[1:]):
        raise RuntimeError("2d test FAILED. Log-likelihood did not monotonically increase")
    else:
        print("2d test PASSED")


def main() -> None:
    print("running tests")
    test_model_1d()
    print()
    test_model_2d()
    print()
    print("tests PASSED. Your model is working")

    # TODO: run your experiments


if __name__ == "__main__":
    main()

