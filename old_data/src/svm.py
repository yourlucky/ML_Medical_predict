import numpy as np
import pandas as pd
import math
import time
import sys
import copy

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing

class SVM:
    def __init__(self, data, x_col, y_col, size):
        self.data = copy.deepcopy(data)
        x = self.data[column]
        self.scaler = preprocessing.StandardScaler().fit(x)
        x = self.scaler.transform(x)
        self.size = size
        self.classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
