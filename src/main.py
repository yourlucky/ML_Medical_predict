from preprocessor import Preprocessor
from regressor import Regressor
from classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
MARGIN = 365*5
CLUSTER_NUM = 7
CLUSTER_COL = 'Group'
SIZE = 0.25
N_NEIGHBORS = 20
DEATH = 2
DEAD = 1
ALIVE = 0

RegressorList = ['linearRegressor', 'knnRegressor', 'nnRegressor']
ClassifierList = ['knnClassifier', 'bayesClassifier', 'svmClassifier', 'nnClassifier']


## Debug purpose
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


def runPreprocessor(path, table_path):
    preprocessor = Preprocessor(path, table_path)
    data = preprocessor.Encode()
    return data

def splitData(data, x_col, y_col):
    _data = shuffle(data)
    _data.reset_index(drop=False, inplace=True)
    idx = int(len(_data) * SIZE)
    test = _data.iloc[:idx]
    train = _data.iloc[idx:]
    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    return train[x_col], train[y_col], test[x_col], test[y_col]

def splitDataColumn(data, x_col, y_col, col):
    _data = shuffle(data)
    _data.reset_index(drop=False, inplace=True)
    
    idx = int(len(data) * SIZE)
    test = data.iloc[:idx]
    train = data.iloc[idx:]
    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    return train[x_col], train[y_col], train[col], test[x_col], test[y_col], test[col]

## deprecated util function    
def _splitData(data ,x_col, y_col):
    given_data = data[data['binary_DEATH [d from CT]'] == DEAD]
    modified_data = data[data['binary_DEATH [d from CT]'] == ALIVE]
    given_data = shuffle(given_data)
    given_data.reset_index(drop=True, inplace=True)
    modified_data = shuffle(modified_data)
    modified_data.reset_index(drop=True, inplace=True)

    idx = int(len(given_data) * SIZE)
    test_data = given_data.iloc[:idx]
    train_data = given_data.iloc[idx:]
    train_data = pd.concat([train_data, modified_data], axis=0)
    train_data.reset_index(drop=True, inplace=True)
    return train_data[x_col], train_data[y_col], test_data[x_col], test_data[y_col]

## deprecated util function
def _splitDataColumn(data ,x_col, y_col, col):
    given_data = data[data['binary_DEATH [d from CT]'] == DEAD]
    modified_data = data[data['binary_DEATH [d from CT]'] == ALIVE]
    given_data = shuffle(given_data)
    given_data.reset_index(drop=True, inplace=True)
    modified_data = shuffle(modified_data)
    modified_data.reset_index(drop=True, inplace=True)

    idx = int(len(given_data) * SIZE)
    test_data = given_data.iloc[:idx]
    train_data = given_data.iloc[idx:]
    train_data = pd.concat([train_data, modified_data], axis=0)
    train_data.reset_index(drop=True, inplace=True)
    return train_data[x_col], train_data[y_col], train_data[col], test_data[x_col], test_data[y_col]


def groupClassify(data, n_group):
        raw_data = np.array((data['_DEATH [d from CT]']))
        unique_data = np.unique(raw_data)
        unique_data = np.sort(unique_data)

        group = np.empty(len(unique_data))
        group_size = len(unique_data) // n_group
        remainder = len(unique_data) % n_group

        idx = 0
        counter = 0
        for g_idx in range(0, n_group):
            size = group_size
            if counter < remainder:
                size += 1
                counter += 1
            for i in range(0, size):
                group[i+idx] = g_idx
            idx += size
            
        g_dict = {}
        for i in range(0, len(unique_data)):
            g_dict[unique_data[i]] = group[i]
        y = np.empty(len(raw_data))
        for i in range(0, len(raw_data)):
            y[i] = g_dict[raw_data[i]]
        _y = pd.DataFrame({CLUSTER_COL: y})
        _data = pd.concat([data, _y], axis=1)
        _data.reset_index(drop=True, inplace=True)
        return _data
        #return _data, g_dict
    
    
def getAccuracy(pred, test, is_regression):
    count = 0
    for i in range(0, len(test)):
        if is_regression == True:
            if abs(pred[i] - test[i]) <= MARGIN:
                count += 1
        else:
            if pred[i] == test.values.tolist()[i]:
                count += 1
    return count / len(test) * 100


## use Regression to predict [d from CT]
def runRegressorTest(x_train, y_train, x_test, y_test):
    print('---------------------------- [d from CT] Regression modeling Tests ------------------------------')
    for reg_arg in RegressorList:
        print('Running ', reg_arg, ' to predict [d from CT] ... ', end='\t')
        reg = Regressor(x_train, y_train, reg_arg)
        reg.fit()

        pred = reg.predict(x_test)
        acc = getAccuracy(pred, y_test, True)
        print('Accuracy: ', acc, '%')

## use Classification to predict [dead/alive]
def runDeathTest(x_train, y_train, x_test, y_test):
    print('---------------------------- [Dead/Alive] modeling Tests ------------------------------')
    for cls_arg in ClassifierList:
        print('Running ', cls_arg, ' to predict [Dead/Alive] ... ', end='\t')
        cls = Classifier(x_train, y_train, DEATH, cls_arg)
        cls.fit()

        pred = cls.predict(x_test)
        acc = getAccuracy(pred, y_test, False)
        print('Accuracy: ', acc, '%')

## use Classification to predict [cluster]
def runClassifierTest(x_train, y_train, x_test, y_test):
    print('---------------------------- [Cluster] modeling Tests ------------------------------')
    for cls_arg in ClassifierList:
        print('Running ', cls_arg, ' to predict [Cluster] ... ', end='\t')
        cls = Classifier(x_train, y_train, CLUSTER_NUM, cls_arg)
        cls.fit()

        pred = cls.predict(x_test)
        acc = getAccuracy(pred, y_test, False)
        print('Accuracy: ', acc, '%')

## use Classification to predict [dead/alive] + use Regression to predict [d from CT]
def runDeathRegressorTest(x_train, y_train, death_train, x_test, y_test):
    print('---------------------------- [Dead/Alive + d from CT] modeling Tests ------------------------------')
    x_col = x_train.columns
    y_col = y_train.name
    death_col = death_train.name
    for cls_arg in ClassifierList:
        print('Running ', cls_arg, ' to predict [Dead/Alive] ... ')
        cls = Classifier(x_train, death_train, DEATH, cls_arg)
        cls.fit()
        for reg_arg in RegressorList:
            print('    Running ', reg_arg, ' to predict [d from CT] ... ', end='\t')
            train = pd.concat([x_train, death_train, y_train], axis=1)
            train.reset_index(drop=True, inplace=True)

            alive_train = train[train[death_col] == ALIVE]
            alive_reg = Regressor(alive_train[x_col], alive_train[y_col], reg_arg)
            alive_reg.fit()
            
            dead_train = train[train[death_col] == DEAD]
            dead_reg = Regressor(dead_train[x_col], dead_train[y_col], reg_arg)
            dead_reg.fit()

            cls_pred = cls.predict(x_test)
            count = 0
            for i in range(0, len(x_test)):
                if cls_pred[i] == ALIVE:
                    y_pred = alive_reg.predict(x_test.iloc[[i]])
                    if abs(y_pred - y_test[i]) <= MARGIN:
                        count += 1
                else:
                    y_pred = dead_reg.predict(x_test.iloc[[i]])
                    if abs(y_pred - y_test[i]) <= MARGIN:
                        count += 1
            acc = count / len(y_test) * 100
            print('Accuracy: ', acc, '%')

## use Classification to predict [cluster] + use Regression to predict [d from CT]
def runClassifierRegressorTest(x_train, y_train, cluster_train, x_test, y_test):
    print('---------------------------- [Cluster + d from CT] modeling Tests ------------------------------')
    x_col = x_train.columns
    y_col = y_train.name
    for cls_arg in ClassifierList:
        print('Running ', cls_arg, ' to predict [Cluster] ... ')
        cls = Classifier(x_train, cluster_train, CLUSTER_NUM, cls_arg)
        cls.fit()
        for reg_arg in RegressorList:
            print('    Running ', reg_arg, ' to predict [d from CT] ... ', end='\t')
            regs = []
            for cluster in range(0, CLUSTER_NUM):
                train = pd.concat([x_train, cluster_train, y_train], axis=1)
                train.reset_index(drop=True, inplace=True)
                train = train[train[CLUSTER_COL] == cluster]
                reg = Regressor(train[x_col], train[y_col], reg_arg)
                reg.fit()
                regs = np.append(regs, reg)

            cls_pred = cls.predict(x_test)
            count = 0
            for i in range(0, len(x_test)):
                y_pred = regs[int(cls_pred[i])].predict(x_test.iloc[[i]])
                if abs(y_pred - y_test[i]) <= MARGIN:
                    count += 1
            acc = count / len(y_test) * 100
            print('Accuracy: ', acc, '%')

## use Classification to predict [dead/alive] + use Classification to predict [cluster] + use Regression to predict [d from CT]
def runDeathClassifierRegressorTest(x_train, y_train, death_train, cluster_train, x_test, y_test):
    print('---------------------------- [Dead/Alive + Cluster + d from CT] modeling Tests ------------------------------')
    x_col = x_train.columns
    y_col = y_train.name
    cluster_col = cluster_train.name
    death_col = death_train.name
    for cls_arg in ClassifierList:
        print('Running ', cls_arg, ' to predict [Dead/Alive] ... ')
        cls = Classifier(x_train, death_train, DEATH, cls_arg)
        cls.fit()
        for _cls_arg in ClassifierList:
            print('    Running ', _cls_arg, ' to predict [Cluster]... ')
            train = pd.concat([x_train, death_train, cluster_train, y_train], axis=1)
            alive_train = train[train[death_col] == ALIVE]
            alive_cls = Classifier(alive_train[x_col], alive_train[cluster_col], CLUSTER_NUM, _cls_arg)
            alive_cls.fit()

            dead_train = train[train[death_col] == DEAD]
            dead_cls = Classifier(dead_train[x_col], dead_train[cluster_col], CLUSTER_NUM, _cls_arg)
            dead_cls.fit()

            for reg_arg in RegressorList:
                print('        Running ', reg_arg, ' to predict [d from CT] ... ', end='\t')
                alive_regs = []
                dead_regs = []

                for cluster in range(0, CLUSTER_NUM):
                    _cluster_train = alive_train[alive_train[cluster_col] == cluster]
                    reg = Regressor(_cluster_train[x_col], _cluster_train[y_col], reg_arg)
                    reg.fit()
                    alive_regs = np.append(alive_regs, reg)
                for cluster in range(0, CLUSTER_NUM):
                    _cluster_train = dead_train[dead_train[cluster_col] == cluster]
                    reg = Regressor(_cluster_train[x_col], _cluster_train[y_col], reg_arg)
                    reg.fit()
                    dead_regs = np.append(dead_regs, reg)

                death_pred = cls.predict(x_test)
                count = 0
                for i in range(0, len(x_test)):
                    if life_pred[i] == ALIVE:
                        cls_pred = alive_cls.predict(x_test.iloc[[i]])
                        y_pred = alive_regs[int(cls_pred)].predict(x_test.iloc[[i]])
                        if abs(y_pred - y_test[i]) <= MARGIN:
                            count += 1
                    else:
                        cls_pred = dead_cls.predict(x_test.iloc[[i]])
                        y_pred = dead_regs[int(cls_pred)].predict(x_test.iloc[[i]])
                        if abs(y_pred - y_test[i]) <= MARGIN:
                            count += 1
                acc = count / len(y_test) * 100
                print('Accuracy: ', acc, '%')



# -------------------------------- Test wrappers -------------------------------------
def RegressorTest(data, x_col, y_col):
    x_train, y_train, x_test, y_test = splitData(data, x_col, y_col)
    runRegressorTest(x_train, y_train, x_test, y_test)
    
def ClassifierRegressorTest(data, x_col, y_col, cluster_col):
    x_train, y_train, x_test, y_test = splitData(data, x_col, y_col)
    train = pd.concat([x_train, y_train], axis=1)
    train = groupClassify(train, CLUSTER_NUM)
    x_train = train[x_col]
    y_train = train[y_col]
    cluster_train = train[cluster_col]

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    cluster_train.reset_index(drop=True, inplace=True)
    runClassifierRegressorTest(x_train, y_train, cluster_train, x_test, y_test)

def DeathRegressorTest(data, x_col, y_col, death_col):
    x_train, y_train, death_train, x_test, y_test, death_test = splitDataColumn(data, x_col, y_col, death_col)
    runDeathRegressorTest(x_train, y_train, death_train, x_test, y_test)

def DeathClassifierRegressorTest(data, x_col, y_col, death_col, cluster_col):
    x_train, y_train, death_train, x_test, y_test, death_test = splitDataColumn(data, x_col, y_col, death_col)
    train = pd.concat([x_train, death_train, y_train], axis=1)
    train = groupClassify(train, CLUSTER_NUM)
    x_train = train[x_col]
    y_train = train[y_col]
    death_train = train[death_col]
    cluster_train = train[cluster_col]

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    death_train.reset_index(drop=True, inplace=True)
    cluster_train.reset_index(drop=True, inplace=True)
    runDeathClassifierRegressorTest(x_train, y_train, death_train, cluster_train, x_test, y_test)
    
def DeathTest(data, x_col, y_col):
    x_train, y_train, x_test, y_test = splitData(data, x_col, y_col)
    runDeathTest(x_train, y_train, x_test, y_test)

def ClassifierTest(data, x_col, y_col):
    _data = groupClassify(data, CLUSTER_NUM)
    x_train, y_train, x_test, y_test = splitData(_data, x_col, y_col)
    runClassifierTest(x_train, y_train, x_test, y_test)


# -------------------------------- Goal wrappers -------------------------------------
def ClinicalOutcome(data):
    print('************************************** Running Goal: Clinical outcome (death) **************************************')
    CT_col = ['L1_HU_BMD', 'TAT Area (cm2)', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    #CT_clinical_col = ['Clinical F/U interval  [d from CT]', 'BMI', 'BMI >30', 'Sex', 'Age at CT', 'Tobacco', 'Alcohol Aggregated', 'FRS 10-year risk (%)', 'FRAX 10y Fx Prob (Orange-w/ DXA)', 'FRAX 10y Hip Fx Prob (Orange-w/ DXA)', 'Met Sx', 'L1_HU_BMD', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    y_col = '_DEATH [d from CT]'
    death_col = 'binary'+y_col
    cluster_col = 'Group'

    _data = data[data[death_col] == DEAD]
    RegressorTest(_data, CT_col, y_col)
    ClassifierRegressorTest(_data, CT_col, y_col, cluster_col)


#    DeathRegressorTest(data, CT_col, y_col, death_col)
#    **** cannot do three-tier gression --> no sample for clusters in ALIVE / DEAD classification
#    DeathClassifierRegressorTest(data, CT_col, y_col, death_col, cluster_col)

def Classification(data):
    print('************************************** Running Goal: Classification **************************************')
    CT_col = ['L1_HU_BMD', 'TAT Area (cm2)', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    #CT_clinical_col = ['Clinical F/U interval  [d from CT]', 'BMI', 'BMI >30', 'Sex', 'Age at CT', 'Tobacco', 'Alcohol Aggregated', 'FRS 10-year risk (%)', 'FRAX 10y Fx Prob (Orange-w/ DXA)', 'FRAX 10y Hip Fx Prob (Orange-w/ DXA)', 'Met Sx', 'L1_HU_BMD', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    y_col = '_DEATH [d from CT]'
    death_col = 'binary'+y_col
    cluster_col = 'Group'

    DeathTest(data, CT_col, death_col)
    ClassifierTest(data, CT_col, cluster_col)
    
def BiologicalAge(data):
    print('************************************** Running Goal: Biological Age **************************************')
    CT_col = ['L1_HU_BMD', 'TAT Area (cm2)', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    #CT_clinical_col = ['Clinical F/U interval  [d from CT]', 'BMI', 'BMI >30', 'Sex', 'Age at CT', 'Tobacco', 'Alcohol Aggregated', 'FRS 10-year risk (%)', 'FRAX 10y Fx Prob (Orange-w/ DXA)', 'FRAX 10y Hip Fx Prob (Orange-w/ DXA)', 'Met Sx', 'L1_HU_BMD', 'Total Body                Area EA (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU', 'L3 SMI (cm2/m2)', 'AoCa        Agatston', 'Liver HU    (Median)']
    y_col = '_DEATH [d from CT]'
    death_col = 'binary'+y_col
    cluster_col = 'Group'

    RegressorTest(data, CT_col, y_col)
    DeathRegressorTest(data, CT_col, y_col, death_col)
    ClassifierRegressorTest(data, CT_col, y_col, cluster_col)

#    **** cannot do three-tier gression --> no sample for clusters in ALIVE / DEAD classification
#    DeathClassifierRegressorTest(data, CT_col, y_col, death_col, cluster_col)

    


    
if __name__ == '__main__':
    data = runPreprocessor('data/OppScrData.csv', 'data/ssa_life_expectancy.csv')
#    Classification(data)
    ClinicalOutcome(data)
#    BiologicalAge(data)   
    

