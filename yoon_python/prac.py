import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
x = np.array([2.0 , 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7])
y = np.array([196, 221, 136, 255, 244, 230, 232, 255, 267])

lr = LinearRegression()

print(x.shape)
x=x.reshape(-1, 1)
print(x.shape)


k=np.array([1,2,3,4])
print(k.shape)

kk=np.array([[1,2,3,4]])
print(kk.shape)
#print(lr.predict([[2.4]]))