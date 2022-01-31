# SYSTEM IMPORTS
import argparse
import numpy as np
import os
from GMM import GMM 
import sys 
sys.path.append('../data')
from load_mnist import load_mnist
from load_musicnet import load_musicnet
from load_imagenet2012 import load_imagenet2012
import sklearn.decomposition
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the data you want to train GMM on")
    parser.add_argument("n_components", type=int, help="Number of clusters you want to train")
    args = parser.parse_args()

    if(args.data == "mnist"):
        mnist = load_mnist("../")
        X = mnist.main()
    elif(args.data == "imagenet"):
        mnist = load_imagenet2012("../", "imagenet2012_500k.npy")
        X = mnist.main()
    elif args.data == "musicnet":
        mnist = load_musicnet("../")
        X = mnist.main()


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

    m.train(X, max_iter=100, monitor_func=collect_ll_per_iter)
    lls = np.array(lls, dtype=float)

    print(lls)

    if np.any(lls[:-1] > lls[1:]):
        raise RuntimeError("MNIST test failed. Log-likelihood did not monotonically increase")

    print("mnist test passed")

if __name__ == "__main__":
    main()



