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

import sys
sys.path.insert(0,'/Users/yoon/Documents/lecture_note/760/medical_project/')
#from preprocessor import Preprocessor

from preprocessor import Preprocessor


def Cluster(_x,cluster_number=7) :
    cluster_calculater= Cluster_made(_x,cluster_number)
    cluster_dict = cluster_calculater.group_dict

    _y=np.empty(len(_x))

    for i in range(0,len(_x)):
        _y[i]=cluster_dict[_x[i]]
    return _y

class Cluster_made:
    def __init__(self, _data,cluster_number=7):
        self.cluster_number=cluster_number
        self._data = _data
        self._data=np.sort(self._data)
        self._data=np.unique(self._data)

        self.group=np.empty(len(self._data))
    
        remain_counter = 0
        index=0

        for group_number in range(1, self.cluster_number+1) :
            correction_share = int (len(self._data)/self.cluster_number)
        
            if remain_counter < (len(self._data) % self.cluster_number) :
                correction_share += 1
                remain_counter += 1 
            for i in range (0, correction_share) :
                self.group[index]= group_number
                index += 1

        self.group_dict ={}

        for i in range(0, len(self._data)):
            self.group_dict[self._data[i]]=self.group[i]


if __name__ == '__main__':

    #_data = Preprocessor('OppScrData.csv','ssa_life_expectancy.csv')
    #data = _data.Encode()
    #data.to_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr.csv', index=True)
    _data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr.csv')
    d_data = np.array((_data['_DEATH [d from CT]']))
    grouping = Cluster(d_data,7)
    data_group = pd.DataFrame({'Group': grouping}) 
    #unique, counts = np.unique(grouping, return_counts=True)
    #print(np.asarray((unique, counts)).T)
    _data=pd.concat([_data,data_group],axis=1)
 
    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    #y_columns = ['Age at CT']
    y_columns = ['Group']

    _x = _data[columns]
    _y = _data[y_columns]
    
    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)
    Y_test.reset_index(drop=True,inplace=True)

    #Model = KNeighborsRegressor(n_neighbors=10)
    #Model = KNeighborsClassifier(n_neighbors=30)
    Model = GaussianNB()
    
    Model.fit(X_train.values, Y_train.values.ravel())
    Predict_value = Model.predict(X_test) #최종 예측 1개의 확률값

    count=0    
    for i in range(0,len(Predict_value)) :
        if abs(Predict_value[i]-Y_test.values[i][0])<1 :
            count+=1


    print("y : {}, Model :{}, accuaracy : {:.2f}%".format(y_columns, Model ,count/len(Predict_value)*100))
    #plt.scatter(_x,_y)
    #plt.show() 
   
