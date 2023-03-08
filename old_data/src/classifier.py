from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class knnClassifier:
    def __init__(self, x, y, n_group):
        self.x = x
        self.y = y
        self.clas = KNeighborsClassifier(n_neighbors=n_group)

    def fit(self):
        self.clas.fit(self.x, self.y.values.ravel())

    def predict(self, x):
        return self.clas.predict(x)

class bayesClassifier:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.clas = GaussianNB()

    def fit(self):
        self.clas.fit(self.x, self.y.values.ravel())

    def predict(self, x):
        return self.clas.predict(x)

class svmClassifier:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scaler = StandardScaler().fit(self.x)
        self.clas = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
#        self.clas = SGDClassifier(loss="log", max_iter=100)

    def fit(self):
        x = self.scaler.transform(self.x)
        self.clas.fit(x, self.y.values.ravel())

    def predict(self, x):
        _x = self.scaler.transform(x)
        return self.clas.predict(_x)
    
class nnClassifier:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.scaler = StandardScaler().fit(self.x)
        self.clas = MLPClassifier(solver='sgd', hidden_layer_sizes=4, learning_rate_init=0.2, max_iter=1000, power_t=0.25, warm_start=True)

    def fit(self):
        x = self.scaler.transform(self.x)
        self.clas.fit(x, self.y.values.ravel())

    def predict(self, x):
        _x = self.scaler.transform(x)
        return self.clas.predict(_x)

class Classifier:
    def __init__(self, x, y, n_group, arg):
        self.arg = arg
        if self.arg == 'knnClassifier':
            self.clas = knnClassifier(x, y, n_group)
        elif self.arg == 'bayesClassifier':
            self.clas = bayesClassifier(x, y)
        elif self.arg == 'svmClassifier':
            self.clas = svmClassifier(x, y)
        elif self.arg == 'nnClassifier':
            self.clas = nnClassifier(x, y)
        else:
            print('---- Not supported Classifier: ', arg)
            exit()

    def fit(self):
        self.clas.fit()

    def predict(self, x):
        return self.clas.predict(x)


