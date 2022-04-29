from distutils.core import setup
from Cython.Build import cythonize
import os


_cd_ = os.path.abspath(os.path.dirname(__file__))
_dbscan_path_ = os.path.join(_cd_, "algs", "DRAM")

setup(name="dbscan", ext_modules=cythonize(os.path.join(_dbscan_path_, 'dbscan.pyx')),)
