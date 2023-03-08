import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing


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

    #_data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_apr.csv')
    _data = pd.read_csv('/Users/yoon/Documents/lecture_note/760/medical_project/yoon_python/data_try_one.csv')
    
    
    d_data = np.array((_data['_DEATH [d from CT]']))
    #grouping = Cluster(d_data,7)
    #data_group = pd.DataFrame({'Group': grouping}) 

    #_data=pd.concat([_data,data_group],axis=1)
    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    #y_columns = ['binary_DEATH [d from CT]']
    #y_columns = ['_DEATH [d from CT]']
    y_columns= ['Made_Group']

    #s_data = _data.loc[_data['binary_DEATH [d from CT]']==1]
    #print(s_data.head)
    #print(s_data['Group'])
   
    _x = _data[columns]
    _y = _data[y_columns]
    print(_y.dtypes)

    #print(_y)

    X_train, X_test, Y_train, Y_test = train_test_split(_x, _y, test_size=0.25)
    Y_test.reset_index(drop=True,inplace=True)

    Model =MLPClassifier(solver='sgd', hidden_layer_sizes=4,
                            learning_rate_init=0.2, max_iter=100,
                            power_t=0.25, warm_start=True)
    

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)
 
 
    Model.fit(X_train, Y_train.values.ravel())
    Predict_value = Model.predict(X_test) #최종 예측 1개의 확률값
 
    count=0    
    for i in range(0,len(Predict_value)) :
        if Predict_value[i]==Y_test.values[i][0] :
            count+=1

    print("y : {}, Model :{}, accuaracy : {:.2f}%".format(y_columns, Model ,count/len(Predict_value)*100))


