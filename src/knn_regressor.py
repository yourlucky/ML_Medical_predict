from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
import copy
N_NEIGHBORS=20

class knnRegressor:
    def __init__(self, data, x_col, y_col, margin, size, neighbors):
        self.x = data[x_col]
        self.y = data[y_col]
        self.margin = margin
        self.size = size
        self.regressor = KNeighborsRegressor(n_neighbors=neighbors)

    def runTest(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.size)
        y_test.reset_index(drop=True, inplace=True)

        self.regressor.fit(x_train, y_train)
        y_pred = self.regressor.predict(x_test)
        correct = 0
        for i in range(0, len(y_test)):
            if abs(y_pred[i] - y_test[i]) <= self.margin:
                correct += 1
        print('Prediction accuracy for Testing data: ', correct/len(y_test)*100, ' %\n')

    def fit(self):
        self.regressor.fit(self.x, self.y)

    def predict(self, x):
        return self.regressor.predict(x)
    
