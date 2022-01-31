# SYSTEM IMPORTS
import argparse
import numpy as np
import os
from gmm import GMM 
import sys 
from datetime import datetime 
sys.path.append('../data')

from data.load_mnist import load as load_mnist
from data.load_imagenet2012 import load as load_imagenet
from data.load_musicnet import load as load_mucisnet

import sklearn.decomposition
from scipy.stats import random_correlation
from typing import List
import kmeans as kmeans_

# PYTHON PROJECT IMPORTS

# CONSTANTS
FILENAME: str = "mnist.npy"


def load(data_path: str) -> np.ndarray:
    if not os.path.exists(data_path):
        raise ValueError("ERROR: file [%s] does not exist" % data_path)
    return  np.load(data_path)


def standardize_data(arr):
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    for column in range(columns):
        mean = np.mean(arr[:,column])
        std = np.std(arr[:,column])
        tempArray = np.empty(0)
        for element in arr[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
        standardizedArray[:,column] = tempArray
    return standardizedArray


def decomposition(X,d):
    covariance_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i/sum(eigen_values))*100)
    # print("variance explained ",variance_explained)
    # cumulative_variance_explained = np.cumsum(variance_explained)
    # print("cumulative variance explained = ",cumulative_variance_explained)    
    projection_matrix = (eigen_vectors.T[:][:d]).T
    X_pca = X.dot(projection_matrix)
    return X_pca 

def data_load(data):
    if(data == "mnist"):
        mnist = load_mnist("../")
        X = mnist.main()
    elif(data == "imagenet"):
        mnist = load_imagenet2012("../", "imagenet2012_500k.npy")
        X = mnist.main()
    elif data == "musicnet":
        mnist = load_musicnet("../")
        X = mnist.main()
    return X
  


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the data you want to train GMM on")
    parser.add_argument("n_components", type=int, help="Number of clusters you want to train")
    args = parser.parse_args()

    dl = datetime.now()
    X = data_load(args.data)
    tdiff_load = datetime.now() - dl 

    print("data load runtime = ", tdiff_load)

    num_features: int = 2
    num_gaussians: int = args.n_components

    X = X[:30000]
    reducer = sklearn.decomposition.PCA(n_components=num_features)
    reducer.fit(X)

    X = standardize_data(reducer.transform(X))

    lls: List[float] = list() 
    def collect_ll_per_iter(m:GMM) -> None:
        lls.append(m.log_likelihood(X))
    
    # X = standardize_data(decomposition(X, num_features).real)
    # print("X shape = ", X.shape)

    m = GMM(num_features, num_gaussians)
    print("init lls: %s" % m.log_likelihood(X))

    tt = datetime.now()
    m.train(X, max_iter=100, monitor_func=collect_ll_per_iter)
    lls = np.array(lls, dtype=float)
    tt_diff = datetime.now() - tt 

    print("model train time = ", tt_diff)
    
    
    # if np.any(lls[:-1] > lls[1:]):
    #     raise RuntimeError("MNIST test failed. Log-likelihood did not monotonically increase")

    # print("mnist test passed")

if __name__ == "__main__":
    srt = datetime.now()
    main()
    srt_diff = datetime.now() - srt 

    print("total script runtime = ", srt_diff)



