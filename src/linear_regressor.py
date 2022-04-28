import numpy as np
import math
import copy
import string
import numbers
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class linearRegressor:
    def __init__(self, x, y):
	self.x = x
        self.y = y
        self.regressor = LinearRegression()

    def fit(self):
        self.regressor.fit(self.x, self.y)

    def predict(self, x):
        return self.regressor.predict(x)
