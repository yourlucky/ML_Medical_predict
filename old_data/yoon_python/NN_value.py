import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing


if __name__ == '__main__':

    #_data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr_handmade.csv')
    _data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr.csv')
    
    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    y_columns = ['_DEATH [d from CT]']

    _x = _data[columns]
    _y = _data[y_columns]
    
    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)
    Y_test.reset_index(drop=True,inplace=True)

    Model = MLPRegressor(hidden_layer_sizes=(16,16,16), activation='relu',learning_rate_init=0.005, solver='adam', max_iter=5000)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)
 
 
    Model.fit(X_train, Y_train.values.ravel())
    Predict_value = Model.predict(X_test) #최종 예측 1개의 확률값
    print("hello")

    count=0    
    for i in range(0,len(Predict_value)) :
        if abs(Predict_value[i]-Y_test.values[i][0]) < 365*3 :
            count+=1

    print("y : {}, Model :{}, accuaracy : {:.2f}%".format(y_columns, Model ,count/len(Predict_value)*100))


