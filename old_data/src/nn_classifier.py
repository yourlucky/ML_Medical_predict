import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing


class nnClassifier:
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

        self.classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=4,
                            learning_rate_init=0.2, max_iter=100,
                            power_t=0.25, warm_start=True)
        self.scaler = preprocessing.StandardScaler().fit(self.x)


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

    def runTest(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.size)
        y_test.reset_index(drop=True, inplace=True)

        x_train = self.scaler.transform(x_train)
        self.fit(x_train, y_train)

        y_pred = self.pred(x_test)
        _y_test = y_test.values.tolist()
        correct = 0
        
        for i in range(0, len(_y_test)):
            if y_pred[i] == _y_test[i]:
                correct += 1

        print('Prediction accuracy for Testing data for Bayes: ', correct/len(y_test)*100, ' %\n')

    def fit(self, x, y):
        self.classifier.fit(x, y.values.ravel())
                
    def pred(self, x):
        _x = self.scaler.transform(x)
        return self.classifier.predict(_x)
