# SYSTEM IMPORTS
from typing import Callable, List
import numpy as np
from scipy.stats import multivariate_normal, random_correlation
# from numpy.random import multivariate_normal

np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS
EPSILON = 1e-12


# This function returns the probability of observing your data given N(mu, cov)
# i.e. how likely it is that N(mu, cov) generated the observed data


class GMM(object):
    def __init__(self, num_features: int, num_gaussians: int) -> None:
        self.num_features: int = int(num_features)
        self.num_gaussians: int = int(num_gaussians)

        # generate one mean vector per gaussian cluster
        self.mus = None 
        self.mus: np.ndarray = np.random.randn(self.num_gaussians, self.num_features)

        # generate one covariance matrix per gaussian cluster, must be positive semidefinite
        self.covs: np.ndarray = None
        if self.num_features > 1:
            # random_correlation.rvs takes in a vector of eigenvalues
            # whose sum must equal the dimensionality of the vector (i.e. 4 eigs must have sum 4, etc)
            random_eigs: np.ndarray = np.random.rand(self.num_gaussians, self.num_features)
            random_eigs /= np.sum(random_eigs, axis=1, keepdims=True)
            random_eigs *= self.num_features
            self.covs = np.stack([random_correlation.rvs(eigs) for eigs in random_eigs], axis=0)
        else:
            # 1-d case....cannot use random_correlation
            self.covs = np.random.rand(self.num_gaussians, self.num_features, self.num_features)

        # init priors to uniform distribution
        self.priors: np.ndarray = np.ones((self.num_gaussians, 1), dtype=float)/self.num_gaussians

    def pdf(self, X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(X, mean=mu, cov=cov)

    def log_likelihood(self, X: np.ndarray) -> float:
        # print(X)
        self.mus = X[np.random.choice(len(X), size=self.num_gaussians, replace=False)]
        
        # print("mus = ",X)
        likelihoods: np.ndarray = np.hstack([self.pdf(X, mu, cov).reshape(-1,1)
                                             for mu,cov in zip(self.mus, self.covs)])
        likelihoods *= self.priors.reshape(1,-1)
        # print(np.sum(likelihoods,axis = 1).shape)
        # print(np.sum((np.sum(likelihoods,axis = 1) == 0)*1) )
        return np.sum(np.log(np.sum(likelihoods, axis=1)))

    def em(self, X: np.ndarray) -> None:
        # TODO: implement this method
        #       This method should perform one iteration of expectation maximization
        """E-step"""
        gamma: np.ndarray = np.hstack([(self.pdf(X, mu.reshape(-1), cov) * prior).reshape(-1, 1)
                                       for mu,cov,prior in zip(self.mus, self.covs, self.priors.reshape(-1))])
        gamma /= (gamma.sum(axis=1, keepdims=True) + EPSILON)

        """M-step"""
        # TODO: update three member variables (attributes, fields, whatever the vocab term you use is):
        #       1) self.priors
        #       2) self.mus
        #       3) self.covs
        self.priors = gamma.mean(axis=0, keepdims=True).T
        self.priors /= self.priors.sum()

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
        Ws: np.ndarray = gamma / gamma.sum(axis=0, keepdims=True)
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


