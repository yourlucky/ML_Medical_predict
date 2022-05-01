import numpy as np
import pandas as pd
import math
import time
import sys
import copy

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing




if __name__ == '__main__':

    #_data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr_handmade.csv')
    _data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr.csv')
    
    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    y_columns = ['binary_DEATH [d from CT]']

    _x = _data[columns]
    _y = _data[y_columns]

    scaler = preprocessing.StandardScaler().fit(_x)
    _x = scaler.transform(_x)
    
    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)
    Y_test.reset_index(drop=True,inplace=True)

    

    #Model = svm.SVC(kernel='linear') //SVM은 데이터가 크면 시간이 오래걸려 못쓴다함
    Model = SGDClassifier(loss="hinge", penalty="l2", max_iter=100) #94.37%
    #Model = SGDClassifier(loss="log", max_iter=100) #94.73%
    
    Model.fit(X_train, Y_train.values.ravel())
    Predict_value = Model.predict(X_test) #최종 예측 1개의 확률값

    count=0    
    for i in range(0,len(Predict_value)) :
        if Predict_value[i]==Y_test.values[i][0] :
            count+=1

    print("y : {}, Model :{}, accuaracy : {:.2f}%".format(y_columns, Model ,count/len(Predict_value)*100))



