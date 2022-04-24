from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import copy

class NnRegressor:
    def __init__(self, data, x_col, y_col, margin, size):
        self.data = copy.deepcopy(data)
        self.x = self.data[x_col]
        self.y = self.data[y_col]
        self.margin = margin
        self.size = size
        self.regressor = MLPRegressor(hidden_layer_sizes=(len(x_col), len(x_col), len(x_col)), max_iter=8000)

    def runTest(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.size)
        y_test.reset_index(drop=True, inplace=True)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        self.regressor.fit(x_train, y_train)
        y_pred = self.regressor.predict(x_test)

        correct = 0
        for i in range(0, len(y_test)):
            if abs(y_pred[i] - y_test[i]) <= self.margin:
                correct += 1
        print("Prediction accuracy for testing data: ", correct / len(y_test) * 100, "%\n")
