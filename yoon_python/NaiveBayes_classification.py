import numpy as np
import pandas as pd
import math
import time
import copy
import string
import numbers
from sklearn.cluster import KMeans, k_means
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score




if __name__ == '__main__':

    #data = pd.read_csv('data_cluster.csv')
    #print(data.dtypes)
    data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_try_one.csv')
    

    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    #y_columns = ['cluster']
    y_columns= ['Made_Group']



    #death_day = ['_DEATH [d from CT]']

    _x = data[columns]
    _y = data[y_columns]

    #https://www.springboard.com/blog/data-analytics/naive-bayes-classification/

    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)

    gnb = GaussianNB()
    gnb.fit(X_train.values, Y_train.values.ravel())
    #https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected

    #Y_gnb_score = gnb.predict_proba(X_test)
    #fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(Y_test, Y_gnb_score[:, 1])

    #sample_data = [[133,686.8,126.0,1.57,26.9,58.2,7367.3,51]]
    #Y_gnb_pred_sample = gnb.predict(sample_data)
    
    Y_gnb_score = gnb.predict_proba(X_test) #1개의 열에 각 클러스터의 7개의 확률값
    Y_gnb_pred = gnb.predict(X_test) #최종 예측 1개의 확률값

    count=0

    for i in range(0,len(Y_gnb_pred)) :
        if Y_gnb_pred[i]==Y_test.values[i][0]:
            count+=1

    print("accuaracy : ", count/len(Y_gnb_pred))

 

    