from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class linearRegressor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.regressor = LinearRegression()

    def fit(self):
        self.regressor.fit(self.x, self.y)

    def predict(self, x):
        return self.regressor.predict(x)

class knnRegressor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.regressor = KNeighborsRegressor(n_neighbors=30)

    def fit(self):
        self.regressor.fit(self.x, self.y)

    def predict(self, x):
        return self.regressor.predict(x)

class nnRegressor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
#        self.regressor = MLPRegressor(hidden_layer_sizes=(len(x.columns), len(x.columns), len(x.columns)), max_iter=8000)
        self.regressor = MLPRegressor(hidden_layer_sizes=(len(x.columns), len(x.columns), len(x.columns)), max_iter=10000)
        self.scaler = StandardScaler().fit(self.x)

    def fit(self):
        x = self.scaler.transform(self.x)
        self.regressor.fit(x, self.y)

    def predict(self, x):
        _x = self.scaler.transform(x)
        return self.regressor.predict(_x)

class Regressor:
    def __init__(self, x, y, arg):
        self.arg = arg
        if arg == 'linearRegressor':
            self.reg = linearRegressor(x, y)
        elif arg == 'knnRegressor':
            self.reg = knnRegressor(x, y)
        elif arg == 'nnRegressor':
            self.reg = nnRegressor(x, y)

    def fit(self):
        self.reg.fit()

    def predict(self, x):
        return self.reg.predict(x)
