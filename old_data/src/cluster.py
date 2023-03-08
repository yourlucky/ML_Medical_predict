import numpy as np
import pandas as pd
import math
import copy

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

N_NEIGHBORS=20

class Cluster:
    def __init__(self, data, x_col, y_col, cluster_num, size):
        self.x = data[x_col]
        self.size = size
        self.cluster_num = cluster_num
        
        raw_data = np.array((data['_DEATH [d from CT]']))
        unique_data = np.unique(raw_data)
        unique_data - np.sort(unique_data)
        
        self.group = self.Group(unique_data, raw_data)
        self.y = pd.DataFrame({y_col: self.group})
        self.y.reset_index(drop=True, inplace=True)
        
        self.knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        self.bayes = GaussianNB()


    def Group(self, unique_data, raw_data):
        group = np.empty(len(unique_data))
        group_size = len(unique_data) // self.cluster_num
        remainder = len(unique_data) % self.cluster_num

        g_dict = {}
        idx = 0
        counter = 0
        for group_idx in range(0, self.cluster_num):
            size = group_size
            if counter < remainder:
                size += 1
                counter += 1

            for i in range(0, size):
                group[i+idx] = group_idx
            idx += size

        g_dict = {}
        for i in range(0, len(unique_data)):
            g_dict[unique_data[i]] = group[i]

        y = np.empty(len(raw_data))
        for i in range(0, len(raw_data)):
            y[i] = g_dict[raw_data[i]]
        return y

    def setGroup(self, data, y_col):
        raw_data = np.array((data['_DEATH [d from CT]']))
        unique_data = np.unique(raw_data)
        unique_data - np.sort(unique_data)

        group = self.Group(unique_data, raw_data)
        y = pd.DataFrame({y_col: group})
        y.reset_index(drop=True, inplace=True)
        _data = pd.concat([data, y], axis=1)
        return _data

    def runKNNTest(self, x_train, x_test, y_train):
        self.knn.fit(x_train, y_train.values.ravel())
        y_pred = self.knn.predict(x_test)
        return y_pred

    def runBayesTest(self, x_train, x_test, y_train):
        self.bayes.fit(x_train, y_train.values.ravel())
        y_pred = self.bayes.predict(x_test)
        return y_pred

    def runTest(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.size)
        y_test.reset_index(drop=True, inplace=True)

        y_predKNN = self.runKNNTest(x_train, x_test, y_train)
        y_predBayes = self.runBayesTest(x_train, x_test, y_train)
        
        _y_test = y_test.values.tolist()
        correct_knn = 0
        correct_bayes = 0
        
        for i in range(0, len(_y_test)):
            if y_predKNN[i] == _y_test[i]:
                correct_knn += 1
            if y_predBayes[i] == _y_test[i]:
                correct_bayes += 1

        print('Prediction accuracy for Testing data for KNN: ', correct_knn/len(y_test)*100, ' %\n')
        print('Prediction accuracy for Testing data for Bayes: ', correct_bayes/len(y_test)*100, ' %\n')

    def fit(self):
        self.bayes.fit(self.x, self.y.values.ravel())
        self.knn.fit(self.x, self.y.values.ravel())
        
    def fitBayes(self, x, y):
        self.bayes.fit(x, y.values.ravel())
        
    def predBayes(self, x):
        return self.bayes.predict(x)

    def fitKNN(self, x, y):
        self.knn.fit(x, y.values.ravel())
        
    def predKNN(self, x):
        return self.knn.predict(x)
    
