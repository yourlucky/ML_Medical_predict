import numpy as np
import pandas as pd
import copy

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn import preprocessing

import group_knn

def _predict(y_columns,type=1) :
    CT_columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    Clincial_columns = ['Clinical F/U interval  [d from CT]', 'BMI', 'BMI >30', 'Sex', 'Age at CT', 'Tobacco', 'Alcohol Aggregated', 'FRS 10-year risk (%)', 'FRAX 10y Fx Prob (Orange-w/ DXA)', 'Met Sx']

    if type ==1 :
        columns = CT_columns + Clincial_columns
    else :
        columns = CT_columns

    _x = _data[columns]
    _y = _data[y_columns]

    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)

    Y_test.reset_index(drop=True,inplace=True)
    X_test.reset_index(drop=True,inplace=True)
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(X_scaled, Y_train.values.ravel())

    Y_pred = model.predict(X_test)

    count=0
    for i in range(0,len(Y_pred)) :
        if abs(Y_pred[i]-Y_test.values[i])<1 :
               count+=1

    print(" {}- {} accuracy {} : {:.2f}%".format(y_columns, type ,model, count/len(Y_pred)*100))




if __name__ == '__main__':
    _data = pd.read_csv('data_bioage.csv')
    y_columns = ['binary_Heart failure DX','category_Heart_failure', 'binary_Type 2 Diabetes DX','category_Type 2 Diabetes DX']

    for i in y_columns :
        _predict(i,1) #type = 1 /x =  CT_columns + Clincial_columns
        _predict(i,0) #type = 0 / x =    X는 CT_columns 



