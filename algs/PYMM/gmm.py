# SYSTEM IMPORTS
from typing import Callable, List
from scipy.stats import random_correlation
import numpy as np
import torch as pt
import torch.distributions.distribution as ptdd
import torch.distributions.multivariate_normal as ptdmvn
import sys
import pymm


np.random.seed(12345)


# PYTHON PROJECT IMPORTS


# CONSTANTS
EPSILON = sys.float_info.epsilon


# This function returns the probability of observing your data given N(mu, cov)
# i.e. how likely it is that N(mu, cov) generated the observed data
def pdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return multivariate_normal.pdf(X, mean=mu, cov=cov, allow_singular=True)


class GMM(object):
    def __init__(self, num_features: int, num_gaussians: int, shelf) -> None:
        self.shelf = shelf
        self.num_features: int = int(num_features)
        self.num_gaussians: int = int(num_gaussians)

        self.shelf.num_features = self.num_features
        self.shelf.num_gaussians = self.num_gaussians

        # generate one mean vector per gaussian cluster
        mus: pt.Tensor = pt.from_numpy(np.random.randn(self.shelf.num_gaussians, self.num_features)).to(pt.float32)

        # generate one covariance matrix per gaussian cluster, must be positive semidefinite
        covs: pt.Tensor = None
        if self.shelf.num_features > 1:
            k: int = int(self.num_features / 2)
            Ws: List[np.ndarray] = [np.random.randn(self.num_features, k) for _ in range(self.num_gaussians)]
            covs = pt.from_numpy(np.stack([np.dot(W, W.T) + np.diag(np.random.randint(low=1, high=self.num_features,
                                                                    size=self.num_features))
                                  for W in Ws], axis=0)).to(pt.float32)
        else:
            # 1-d case....cannot use random_correlation
            covs = pt.from_numpy(np.random.rand(self.num_gaussians, self.num_features, self.num_features))\
                .to(pt.float32)
        self.shelf.means = mus
        self.shelf.covs = covs

        self.shelf.likelihoods = None
        self.shelf.gammas = None
        self.shelf.Ws = None

        self.pt_gaussian_objs: List[ptdd.Distribution] = [ptdmvn.MultivariateNormal(mu, scale_tril=pt.tril(cov))
                                                          for mu, cov in zip(mus, covs)]

        # init priors to uniform distribution
        self.shelf.priors: pt.Tensor = pt.ones((self.num_gaussians, 1), dtype=pt.float32)/self.num_gaussians
        self.shelf.gpu_id: int = None
        self.gpu_id = None

        self.priors = self.shelf.priors

    def to(self, gpu_id: int = None) -> "GMM":
        gpu_id = "cpu" if gpu_id is None else gpu_id
        self.pt_gaussian_objs = [ptdmvn.MultivariateNormal(obj.mean.to(gpu_id),
                                                           scale_tril=obj.scale_tril.to(gpu_id))
                                 for obj in self.pt_gaussian_objs]
        self.priors = self.priors.to(gpu_id)
        self.shelf.gpu_id = gpu_id
        self.gpu_id = gpu_id
        return self

    def batch_likelihood(self, X: pt.Tensor, batch_size: int = None) -> pt.Tensor:
        X_device: int = "cpu" if not X.is_cuda else X.get_device()
        X_on_same_device: bool = X_device == self.gpu_id
        batch_size: int = X.size(0) if batch_size is None else batch_size

        likelihoods: pt.Tensor = pt.empty((X.size(0), self.num_gaussians), dtype=pt.float32, device=self.gpu_id)

        if self.shelf.likelihoods is None or self.shelf.likelihoods.size() != likelihoods.size():
            self.shelf.likelihoods = likelihoods.cpu()
        else:
            self.shelf.likelihoods.copy_(likelihoods)
        self.shelf.likelihoods.persist()

        if batch_size == X.size(0):
            for cluster_idx in range(self.num_gaussians):
                pt.exp(self.pt_gaussian_objs[cluster_idx].log_prob(X), out=likelihoods[:, cluster_idx])
                self.shelf.likelihoods[:, cluster_idx].copy_(likelihoods[:, cluster_idx])
                self.shelf.likelihoods.persist()
        else:
            num_batches: int = int(np.ceil(X.size(0) / batch_size))
            X_batch_buffer: pt.Tensor = pt.empty((batch_size, X.size(1)), dtype=X.dtype, device=self.gpu_id)

            # print("batching....num_batches", num_batches)

            for batch_idx in range(num_batches-1):
                X_batch_buffer.copy_(X[batch_idx*batch_size: (batch_idx+1)*batch_size, :])
                for cluster_idx in range(self.num_gaussians):
                    pt.exp(self.pt_gaussian_objs[cluster_idx].log_prob(X_batch_buffer),
                           out=likelihoods[batch_idx*batch_size: (batch_idx+1)*batch_size, cluster_idx])
                    self.shelf.likelihoods[batch_idx*batch_size: (batch_idx+1)*batch_size, cluster_idx].copy_(
                        likelihoods[batch_idx*batch_size: (batch_idx+1)*batch_size, cluster_idx])
                    self.shelf.likelihoods.persist()

            # now do the last batch
            batch_size = min(num_batches*batch_size, X.size(0)) - (num_batches-1)*batch_size
            X_batch_buffer = X[-batch_size:, :].to(self.gpu_id)
            for cluster_idx in range(self.num_gaussians):
                pt.exp(self.pt_gaussian_objs[cluster_idx].log_prob(X_batch_buffer),
                           out=likelihoods[-batch_size:, cluster_idx])
                self.shelf.likelihoods[-batch_size:, cluster_idx].copy_(likelihoods[-batch_size:, cluster_idx])
                self.shelf.likelihoods.persist()

        likelihoods *= self.priors.T
        self.shelf.likelihoods.copy_(likelihoods)
        self.shelf.likelihoods.persist()
        return likelihoods

    def log_likelihood(self, X: pt.Tensor, batch_size: int = None) -> float:
        # likelihoods: pt.Tensor = pt.empty((X.size(0), self.num_gaussians), dtype=pt.float32)
        # for cluster_idx in range(self.num_gaussians):
        #     likelihoods[:, cluster_idx] = pt.exp(self.pt_gaussian_objs[cluster_idx].log_prob(X))
        # likelihoods *= self.priors.T

        # return pt.sum(pt.log(pt.sum(likelihoods, dim=1) + EPSILON))

        return pt.sum(pt.log(pt.sum(self.batch_likelihood(X, batch_size=batch_size), dim=1) + EPSILON))

    def m_step(self, X: pt.Tensor, Ws: pt.Tensor, batch_size=None) -> None:
        X_device: int = "cpu" if not X.is_cuda else X.get_device()
        X_on_same_device: bool = X_device == self.gpu_id

        batch_size: int = X.size(0) if batch_size is None else batch_size
        num_batches: int = int(np.ceil(X.size(0) / batch_size))

        # print("batching....num_batches", num_batches)

        mus: pt.Tensor = pt.zeros((self.num_gaussians, self.num_features),
                                  dtype=pt.float32, device=self.gpu_id)
        covs: pt.Tensor = pt.zeros((self.num_gaussians, self.num_features, self.num_features),
                                   dtype=pt.float32, device=self.gpu_id)

        batch_covs_buffer: pt.Tensor = pt.empty((batch_size, self.num_features, self.num_features),
                                                dtype=pt.float32, device=self.gpu_id)
        cov_aggregate_buffer: pt.Tensor = pt.empty((self.num_features, self.num_features),
                                                   dtype=pt.float32, device=self.gpu_id)
        mu_aggregate_buffer: pt.Tensor = pt.empty((self.num_features,), dtype=pt.float32, device=self.gpu_id)
        X_batch_buffer: pt.Tensor = pt.empty((batch_size, self.num_features),
                                             dtype=pt.float32, device=self.gpu_id)
        X_minus_mu_batch_buffer: pt.Tensor = pt.empty_like(X_batch_buffer)

        for batch_idx in range(num_batches-1):

            for cluster_idx in range(self.num_gaussians):
                pt.sub(X_batch_buffer, self.pt_gaussian_objs[cluster_idx].mean, out=X_minus_mu_batch_buffer)
                Ws_batch: pt.Tensor = Ws[-batch_size:,
                                         cluster_idx:cluster_idx+1, None]

                pt.sum(Ws_batch.view(-1,1) * X_batch_buffer, dim=0, out=mu_aggregate_buffer)
                mus[cluster_idx, :].add_(mu_aggregate_buffer)
                self.shelf.means[cluster_idx,:].copy_(mus[cluster_idx,:])
                self.shelf.means.persist()

                pt.sub(X_batch_buffer, self.pt_gaussian_objs[cluster_idx].mean, out=X_minus_mu_batch_buffer)

                # use batch to update cov
                pt.matmul(X_minus_mu_batch_buffer[:,:,None], X_minus_mu_batch_buffer[:,None,:], out=batch_covs_buffer)
                batch_covs_buffer.mul_(Ws_batch)
                covs[cluster_idx, :, :].add_(pt.sum(batch_covs_buffer, out=cov_aggregate_buffer,
                                                    dim=0))
                self.shelf.covs[cluster_idx,:,:].copy_(covs[cluster_idx,:,:])
                self.shelf.covs.persist()

        # do the last batch
        batch_size: int = min(num_batches*batch_size, X.size(0)) - (num_batches-1)*batch_size
        batch_covs_buffer = batch_covs_buffer[:batch_size, :, :]
        X_batch_buffer = X_batch_buffer[:batch_size, :]
        X_batch_buffer.copy_(X[-batch_size:, :])
        X_minus_mu_batch_buffer = X_minus_mu_batch_buffer[-batch_size:, :]

        for cluster_idx in range(self.num_gaussians):
            pt.sub(X_batch_buffer, self.pt_gaussian_objs[cluster_idx].mean, out=X_minus_mu_batch_buffer)
            Ws_batch: pt.Tensor = Ws[-batch_size:,
                                     cluster_idx:cluster_idx+1, None]

            pt.sum(Ws_batch.view(-1,1) * X_batch_buffer, dim=0, out=mu_aggregate_buffer)
            mus[cluster_idx, :].add_(mu_aggregate_buffer)
            self.shelf.means[cluster_idx,:].copy_(mus[cluster_idx,:])
            self.shelf.means.persist()

            pt.sub(X_batch_buffer, self.pt_gaussian_objs[cluster_idx].mean, out=X_minus_mu_batch_buffer)

            # use batch to update cov
            pt.matmul(X_minus_mu_batch_buffer[:,:,None], X_minus_mu_batch_buffer[:,None,:], out=batch_covs_buffer)
            batch_covs_buffer.mul_(Ws_batch)
            covs[cluster_idx, :, :].add_(pt.sum(batch_covs_buffer, out=cov_aggregate_buffer,
                                                dim=0))
            self.shelf.covs[cluster_idx,:,:].copy_(covs[cluster_idx,:,:])
            self.shelf.covs.persist()

        mus.add_(EPSILON)
        diagonal_byte_mask: pt.Tensor = pt.eye(self.num_features).bool()
        for cluster_idx, (mu, cov) in enumerate(zip(mus, covs)):
            # print(mu, cov)
            cov[diagonal_byte_mask] = pt.diag(cov) + EPSILON # hack to keep large dim covs positive semi-definite
            self.shelf.covs[cluster_idx,:,:].copy_(cov)
            self.shelf.covs.persist()
            self.pt_gaussian_objs[cluster_idx] = ptdmvn.MultivariateNormal(mu,
                                                                           covariance_matrix=cov)

    def em(self, X: pt.Tensor, batch_size: int = None) -> None:
        # TODO: implement this method
        #       This method should perform one iteration of expectation maximization
        """E-step"""
        gamma: pt.Tensor = self.batch_likelihood(X, batch_size=batch_size)
        # gamma: np.ndarray = np.hstack([(pdf(X, mu.reshape(-1), cov) * prior).reshape(-1, 1)
        #                                for mu,cov,prior in zip(self.mus, self.covs, self.priors.reshape(-1))])

        if self.shelf.gamma is None or self.shelf.gamma.size() != self.shelf.likelihoods.size():
            self.shelf.gamma = gamma.cpu()
        else:
            self.shelf.gamma.copy_(gamma)
        self.shelf.gamma.persist()

        gamma /= (gamma.sum(dim=1, keepdims=True) + EPSILON)
        self.shelf.gamma.copy_(gamma)
        self.shelf.gamma.persist()

        """M-step"""
        # TODO: update three member variables (attributes, fields, whatever the vocab term you use is):
        #       1) self.priors
        #       2) self.mus
        #       3) self.covs
        self.priors = gamma.mean(dim=0, keepdims=True).T
        self.priors /= (self.priors.sum() + EPSILON)
        self.shelf.priors.copy_(self.priors)
        self.shelf.priors.persist()

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
        Ws: pt.Tensor = gamma / (gamma.sum(dim=0, keepdims=True) + EPSILON)
        if self.shelf.Ws is None or self.shelf.Ws.size() != Ws.size():
            self.shelf.Ws = Ws.cpu()
        else:
            self.shelf.Ws.copy_(Ws)
        self.shelf.Ws.persist()
        # self.mus = np.vstack([np.sum(Ws[:,k].reshape(-1,1)*X, axis=0, keepdims=True) for k in range(self.num_gaussians)])
        # self.covs = np.stack([sum(Ws[i,k]*(x_i - mu).reshape(-1,1).dot((x_i - mu).reshape(1,-1))
        #                           for i,x_i in enumerate(X))
        #                       for k, mu in enumerate(self.mus)], axis=0)

        self.m_step(X, Ws, batch_size=batch_size)

    def train(self, X: np.ndarray, epsilon: float = 1e-9, max_iter: int = 1e6, batch_size: int = None,
              monitor_func: Callable[["GMM"], None] = None) -> None:
        # print(batch_size)

        # structure of iterative algorithm: while (dont give up) and (havent converged): do stuff
        current_iter: int = 0
        prev_ll: float = sys.float_info.max
        current_ll: float = sys.float_info.epsilon

        while current_iter < max_iter and abs(prev_ll - current_ll)/abs(prev_ll) > epsilon:
            self.em(X, batch_size=batch_size)
            

            prev_ll = current_ll
            current_ll = self.log_likelihood(X, batch_size=batch_size)
            current_iter += 1

            if monitor_func is not None:
                monitor_func(self)


def test_model_1d() -> None:
    print("running 1d test")
    num_samples: int = 100
    num_gaussians: int = 3

    real_mus: np.ndarray = np.array([-4, 4, 0], dtype=float)
    real_vars: np.ndarray = np.array([1.2, 0.8, 2], dtype=float)

    X: np.ndarray = pt.from_numpy(np.vstack([np.random.normal(loc=rmu, scale=rvar, size=num_samples).reshape(-1,1)
                               for rmu, rvar in zip(real_mus, real_vars)])).to(pt.float32)

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

    X: np.ndarray = pt.from_numpy(np.vstack([np.random.multivariate_normal(rmu, rcov, size=num_samples)
                              for rmu, rcov in zip(real_mus, real_covs)])).to(pt.float32)

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
        print(lls, lls[:-1] > lls[1:])
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
