# SYSTEM IMPORTS
from typing import Callable, List
from scipy.stats import multivariate_normal, random_correlation
import numpy as np


np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS
EPSILON = 1e-12


# This function returns the probability of observing your data given N(mu, cov)
# i.e. how likely it is that N(mu, cov) generated the observed data
def pdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return multivariate_normal.pdf(X, mean=mu, cov=cov, allow_singular=True)


class GMM(object):
    def __init__(self, num_features: int, num_gaussians: int) -> None:
        self.num_features: int = int(num_features)
        self.num_gaussians: int = int(num_gaussians)

        # generate one mean vector per gaussian cluster
        self.mus: np.ndarray = np.random.randn(self.num_gaussians, self.num_features)

        # generate one covariance matrix per gaussian cluster, must be positive semidefinite
        self.covs: np.ndarray = None
        if self.num_features > 1:
            k: int = int(self.num_features / 2)
            Ws: List[np.ndarray] = [np.random.randn(self.num_features, k) for _ in range(self.num_gaussians)]
            self.covs = np.stack([np.dot(W, W.T) + np.diag(np.random.randint(low=1, high=self.num_features,
                                                                             size=self.num_features))
                                  for W in Ws], axis=0)
        else:
            # 1-d case....cannot use random_correlation
            self.covs = np.random.rand(self.num_gaussians, self.num_features, self.num_features)

        # init priors to uniform distribution
        self.priors: np.ndarray = np.ones((self.num_gaussians, 1), dtype=float)/self.num_gaussians

    def log_likelihood(self, X: np.ndarray) -> float:
        likelihoods: np.ndarray = np.hstack([pdf(X, mu, cov).reshape(-1,1)
                                             for mu,cov in zip(self.mus, self.covs)])
        likelihoods *= self.priors.reshape(1,-1)
        return np.sum(np.log(np.sum(likelihoods, axis=1) + EPSILON))

    def em(self, X: np.ndarray) -> None:
        # TODO: implement this method
        #       This method should perform one iteration of expectation maximization
        """E-step"""
        gamma: np.ndarray = np.hstack([(pdf(X, mu.reshape(-1), cov) * prior).reshape(-1, 1)
                                       for mu,cov,prior in zip(self.mus, self.covs, self.priors.reshape(-1))])
        gamma /= (gamma.sum(axis=1, keepdims=True) + EPSILON)

        """M-step"""
        # TODO: update three member variables (attributes, fields, whatever the vocab term you use is):
        #       1) self.priors
        #       2) self.mus
        #       3) self.covs
        self.priors = gamma.mean(axis=0, keepdims=True).T
        self.priors /= (self.priors.sum() + EPSILON)

        """We can do this at least two ways. The first way (commented out) is more direct but less efficient:
           This way computes the formulae (from question 2) as they are written.
        """
        # self.mus = np.vstack([np.sum(gamma[:,k].reshape(-1,1) * X, axis=0, keepdims=True) / gamma[:,k].sum()
        #                       for k in range(self.num_gaussians)])
        # self.covs = np.stack([sum([gamma_ik*(x_i - mu).reshape(-1,1).dot((x_i - mu).reshape(-1,1).T)
        #                               for x_i, gamma_ik in zip(X, gamma[:,k])]) / gamma[:,k].sum()
        #                       for k,mu in enumerate(self.mus)], axis=0)

        """This is the second way to do it. In this approach, we recognize that computing the MLE estimates for
           each mu and Sigma involves computing gamma_{ij} / \sum_k gamma_{kj}. If we precompute these values
           (as a matrix), we can save a lot of time (for large data matrices)
        """
        Ws: np.ndarray = gamma / (gamma.sum(axis=0, keepdims=True) + EPSILON)
        self.mus = np.vstack([np.sum(Ws[:,k].reshape(-1,1)*X, axis=0, keepdims=True) for k in range(self.num_gaussians)])
        self.covs = np.stack([sum(Ws[i,k]*(x_i - mu).reshape(-1,1).dot((x_i - mu).reshape(1,-1))
                                  for i,x_i in enumerate(X))
                              for k, mu in enumerate(self.mus)], axis=0)

    def train(self, X: np.ndarray, epsilon: float = 1e-9, max_iter: int = 1e6,
              monitor_func: Callable[["GMM"], None] = None) -> None:

        # structure of iterative algorithm: while (dont give up) and (havent converged): do stuff
        current_iter: int = 0
        prev_ll: float = np.inf
        current_ll: float = 0.0

        while current_iter < max_iter and abs(prev_ll - current_ll) > epsilon:
            self.em(X)
            

            prev_ll = current_ll
            current_ll = self.log_likelihood(X)
            current_iter += 1

            if monitor_func is not None:
                monitor_func(self)


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
                              for rmu, rcov in zip(real_mus, real_covs)])

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
