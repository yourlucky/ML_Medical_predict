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

def Bio_answer(_data) :
    # excluding someone who dided in 3yrs ('_death_in_3yr'==1)
    valid_data = _data[_data['_death_in_3yr']==0]
    valid_data.reset_index(drop=True, inplace=True)
    valid_data = valid_data.groupby(['I_age'],as_index=False).mean()

    Test_set = copy.deepcopy(valid_data)

    return Test_set



if __name__ == '__main__':
    #전체 데이터 순서
    # 1) 평균값을 통한 test set 만들기
    # 2) 단순 regression 모델로 맞춰보기
    # 3) 2등분 클러스터 생성하기
    # 4) 2등분 클러스터에 대해 각각 Regression 돌려보기


    #_data = pd.read_csv('data_apr_handmade.csv')
    #기본적 데이터로드
    _data = pd.read_csv('data_bioage.csv')
    
    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    y_columns = ['I_age']
    
    #평균 testset 만들기
    _test=Bio_answer(_data)
   
    _x = _data[columns]
    _y = _data[y_columns]

    test_x= _test[columns]
    y_test=_test['I_age']

    # # 2단계 
    train_x =_data[_data['binary_DEATH [d from CT]']==1]
    
    # #모델 생성
    # # 1단계 단순모델로 맞춰보기
    #model = KNeighborsRegressor(n_neighbors=5) #3년 28.5% 5년 40%
    #model = GaussianNB() # 1년 -42%, 2년 - 45%, 3년 51%, 5년 -60%
    model = MLPRegressor(hidden_layer_sizes=(10,15,3), activation='relu',learning_rate_init=0.001, solver='adam', max_iter=5000) # 3년 27%, 5년 39%

    scaler = preprocessing.StandardScaler().fit(_x)

    X_scaled = scaler.transform(_x)
    model.fit(X_scaled, _y.values.ravel())
    
    test_x_scaled = scaler.transform(test_x)
    Y_pred = model.predict(test_x_scaled)

    # count=0
    # for i in range(0,len(Y_pred)) :
    #     if abs(Y_pred[i]-y_test[i])<=3 :
    #         count+=1

    # print("1단계 {} : {:.2f}%".format(model, count/len(Y_pred)*100))

    #2단계 클러스터 2개 그룹 해보기

    # # 클러스터 2개 만들고 붙이는 부분
    g_data = pd.read_csv('data_bioage.csv')
    g_data = g_data[g_data['binary_DEATH [d from CT]']==0]
    g_data.reset_index(drop=True,inplace=True)

    gd_data = np.array((g_data['_DEATH [d from CT]']))
    grouping = group_knn.Cluster(gd_data,2)
    data_group = pd.DataFrame({'Group': grouping}) 
    g_data=pd.concat([g_data,data_group],axis=1)

    #클러스터의 예측정확도 잡기
    #Model = KNeighborsRegressor(n_neighbors=10) #96.18%
    #Model = KNeighborsClassifier(n_neighbors=10) #96.62
    Model = GaussianNB() #65.54

    #Model =MLPClassifier(solver='sgd', hidden_layer_sizes=(20,20,5),  #96.90
    #                      learning_rate_init=0.001, max_iter=10000, activation='relu',power_t=0.25) #66.82%

    g_x = g_data[columns]
    g_y = g_data['Group']
    
    gX_train, gX_test, gY_train, gY_test = train_test_split(g_x, g_y, test_size=0.25)
    gY_test.reset_index(drop=True,inplace=True)
    scaler = preprocessing.StandardScaler().fit(gX_train)
    gX_scaled = scaler.transform(gX_train)
    gX_test = scaler.transform(gX_test)
    
    Model.fit(gX_train.values, gY_train.values.ravel())
    # Predict_value = Model.predict(gX_test) #최종 예측 1개의 확률값

    # count=0    
    # for i in range(0,len(Predict_value)) :
    #      if abs(Predict_value[i]-gY_test[i])<1:
    #          count+=1

    # print("Grouping :{}, accuaracy : {:.2f}%".format(Model ,count/len(Predict_value)*100))

    # #예측한 그룹값을 붙인다.
    #Group_Predict_value = Model.predict(g_x)
    #Gruop_Predict= pd.DataFrame({'Predict_Group':Group_Predict_value})
    #Group_P=pd.concat([g_data,Gruop_Predict],axis=1)

    # #3-1단계 2등분 클러스터에 대해 각각 리그레션 해보기
    #g_data.to_csv('data_group.csv')

    g_data = pd.read_csv('data_group.csv')

    # #평균 testset 만들기
    _test=Bio_answer(g_data)
   
    test_x= _test[columns]
    y_test=_test['I_age']

    #예측한 그룹이 1 인것과 2인것을 각각 생성
    G_data_one = g_data[g_data['Group']==1]
    G_data_two = g_data[g_data['Group']==2]
    G_data_one.reset_index(drop=True,inplace=True)
    G_data_two.reset_index(drop=True,inplace=True)
   
    GF_x_one = G_data_one[columns]
    GF_y_one = G_data_one[y_columns]
    
    GF_x_two = G_data_two[columns]
    GF_y_two = G_data_two[y_columns]

    #model_one = KNeighborsRegressor(n_neighbors=5) #3년 28.5% 5년 40%
    #model_two = KNeighborsRegressor(n_neighbors=5) #3년 28.5% 5년 40%
    #model_one = GaussianNB() # 1년 -42%, 2년 - 45%, 3년 51%, 5년 -60%
    #model_two = GaussianNB()
    #model_one = MLPRegressor(hidden_layer_sizes=(20), activation='relu',learning_rate_init=0.001, solver='adam', max_iter=5000) # 3년 27%, 5년 39%
    #model_two = MLPRegressor(hidden_layer_sizes=(20), activation='relu',learning_rate_init=0.001, solver='adam', max_iter=5000) # 3년 27%, 5년 39%
    model_one =MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=10000)
    model_two =MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=10000)


    scaler_one = preprocessing.StandardScaler().fit(GF_x_one)
    X_scaled_one = scaler_one.transform(GF_x_one)

    #scaler_two = preprocessing.StandardScaler().fit(GF_x_two)
    #X_scaled_two = scaler_one.transform(GF_x_two)

    model_one.fit(GF_x_one, GF_y_one.values.ravel())
    model_two.fit(GF_x_two, GF_y_two.values.ravel())

    Predict_value = Model.predict(test_x) #그룹값 예측
    
    Y_pred_one = model_one.predict(test_x)
    Y_pred_two = model_two.predict(test_x)
    
    y_test_numpy=y_test.to_numpy()
    k_=np.concatenate([y_test_numpy,Predict_value])
    #print(k_)

    
    count=0
    for i in range(0,len(Y_pred_one)) :
       if Predict_value[i] == 2 :
           if abs(Y_pred_two[i]-y_test[i]) <=3 :
               count +=1
       else :
           if abs(Y_pred_one[i]-y_test[i])<=3 :
               count+=1

    print("2 step {} : {:.2f}%".format(model_one, count/len(Y_pred_one)*100))
