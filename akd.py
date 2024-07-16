import sys

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

import itertools

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from .ncf import ncf
from .ncf.dataset import Dataset as NCFDataset

import warnings
warnings.filterwarnings('ignore')

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))