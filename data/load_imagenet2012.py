# SYSTEM IMPORTS
from typing import Dict, List, Tuple
import argparse
import numpy as np
import os
import requests


# PYTHON PROJECT IMPORTS



# CONSTANTS



class load_imagenet2012:
    def __init__(self, path, filename):
        self.data_path = path 
        self.FILENAME = filename
        self.FILE_CHOICES = ["imagenet2012_500k.npy", "imagenet2012_800k.npy",]
        if filename not in self.FILE_CHOICES:
            return ValueError("enter one of the flies among " + str(self.FILE_CHOICES))
        self.main()


    def load(self,filepath) -> np.ndarray:
        if not os.path.exists(filepath):
            raise ValueError("ERROR: file [%s] does not exist" % filepath)

        data_dict: Dict[str, np.ndarray] = np.load(filepath)
        X: np.ndarray = np.vstack([data_dict["X_train"], data_dict["X_test"]])
        return X.reshape(X.shape[0], -1)


    def main(self) -> np.ndarray:
        mnist_filepath: str = os.path.join(self.data_path, self.FILENAME)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        return self.load(mnist_filepath)