import numpy as np
import pandas as pd
import math
import time
import sys
import copy
import string
import numbers

COL_CNT = 52
FIRST_CT_COL_IDX = 41

class Preprocessor:
    def __init__(self, path, table_path):
        self.drop_num = 38
        self.data = self.ReadInput(path)
	# lookup table has the life expectancy of people in sex and ages
	# (reference: https://www.ssa.gov/oact/STATS/table4c6.html)
        self.lookup_tab = self.ReadInput(table_path)
    
    def ReadInput(self, path):
        df = pd.read_csv(path)
        return df
    
    def BMI(self, _data):
        data = copy.deepcopy(_data)
        for i in range(0, len(data)):
            if not data[i]:
                data[i] = data.mean()
        return data
    
    def BMIrange(self, _data):
        data = copy.deepcopy(_data)
        for i in range(0, len(data)):
            if data[i] == 'Y':
                data[i] = 1
            elif data[i] == 'N':
                data[i] = 0
            else:
                data[i] = 0.5
        return data
    
    def Sex(self, _data):
        data = copy.deepcopy(_data)
        for i in range(0, len(data)):
            if data[i] == 'Male':
                data[i] = 1
            else:
                data[i] = 0
        return data

    def Tobacco(self, _data):
        data = copy.deepcopy(_data)
        for i in range(0, len(data)):
            if data[i] == 'Yes':
                data[i] = 1
            elif data[i] == 'No':
                data[i] = 0
            else:
                data[i] = 0.5
        return data
    
    def Alcohol(self, _data, column):
        data = copy.deepcopy(_data)
        null = data[column].str.isspace()
        dict_ = {}
        for i in range(0, len(data)):
            if null[i] == False:
                string = data.at[i, column]
                val_list = string.split(",")
                for val in val_list:
                    if dict_.get(val) == None:
                        dict_[val] = 1
                        data[val] = 0
        data['Alcohol Aggregated'] = 0
        for i in range(0, len(data)):
            if null[i] == False:
                string = data.at[i, column]
                val_list = string.split(",")
                for val in val_list:
                    data.at[i, val] = 1
                    data.at[i, 'Alcohol Aggregated'] = 1
        
        data = data.drop(column, axis=1)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def predictFRS(self, _data):
        columns = ['BMI >30', 'Age at CT', 'FRAX 10y Fx Prob (Orange-w/ DXA)', 'Sex', 'Tobacco']
        theta = [2.877e-01, 2.914e-01,  -1.442e-01, 5.038e+00,  1.949e+00]
        data = _data[columns]
        
        y_col = 'FRS 10-year risk (%)'
        x = data.to_numpy()
#         x = np.ones((data.to_numpy().shape[0], 1))
#         x = np.concatenate((x, data.to_numpy()), axis=1)
        x = x.astype(np.float)
        datax = _data[y_col].str.contains('X')
        for i in range(0, len(datax)):
            if datax[i] == True:
                y = np.dot(x[i], theta) - 1.52e+01
                _data.at[i, y_col] = y
        return _data
        
    def FRS(self, _data):
        data = copy.deepcopy(_data)
        data = data.str.replace('%', '')
        less = data.str.contains('<')
        bigger = data.str.contains('>')
        numeric = data.str.isnumeric()
        cnt = total = 0
        for i in range(0, len(data)):
            if numeric[i] == True:
                total += float(data[i])
                cnt += 1
            elif less[i] == True:
                data[i] = 0.5
                total += 0.5
                cnt += 1
            elif bigger[i] == True:
                data[i] = 65
                total += 65
                cnt += 1
        return data
    
    def predictFRAX_Fx(self, _data):
        columns = ['BMI', 'Age at CT','FRS 10-year risk (%)','Alcohol Aggregated', 'Sex', 'Tobacco']
        theta = [-0.10778, 0.34435, -0.14684, 1.17095, -1.51354, 0.46751]
        data = _data[columns]
        
        y_col = 'FRAX 10y Fx Prob (Orange-w/ DXA)'
        x = data.to_numpy()
#         x = np.ones((data.to_numpy().shape[0], 1))
#         x = np.concatenate((x, data.to_numpy()), axis=1)
        x = x.astype(np.float)
        data_ = _data[y_col].str.contains('-1')
        for i in range(0, len(data_)):
            if data_[i] == True:
                y = np.dot(x[i], theta) - 9.91132
                _data.at[i, y_col] = y
        return _data
    
    def predictFRAX_Hip(self, _data):
        columns = ['BMI', 'Age at CT','Alcohol Aggregated', 'Sex', 'Tobacco']
        theta = [-0.04757,0.13513,0.38748,-0.38907,0.17181]
        data = _data[columns]
        
        y_col = 'FRAX 10y Fx Prob (Orange-w/ DXA)'
        x = data.to_numpy()
#         x = np.ones((data.to_numpy().shape[0], 1))
#         x = np.concatenate((x, data.to_numpy()), axis=1)
        x = x.astype(np.float)
        data_ = _data[y_col].str.contains('-1')
        for i in range(0, len(data_)):
            if data_[i] == True:
                y = np.dot(x[i], theta) - 5.43647
                _data.at[i, y_col] = y
        return _data
    
    def FRAX(self, _data):
        data = copy.deepcopy(_data)
        underscore = data.str.contains('_')
        cnt = total = 0
        for i in range(0, len(data)):
            if underscore[i] == False:
                cnt += 1
                total += float(data[i])
        avg = total / cnt
        for i in range(0, len(data)):
            if underscore[i] == True:
                data[i] = -1
        return data
        
    def MetSx(self, _data):
        data = copy.deepcopy(_data)
        for i in range(0, len(data)):
            if data[i] == 'Y':
                data[i] = 1
            elif data[i] == 'N':
                data[i] = 0
            else:
                data[i] = 0.5
        return data
    
    def Drop(self, _data):
        data = copy.deepcopy(_data)
        col = ['FRS 10-year risk (%)', 'FRAX 10y Fx Prob (Orange-w/ DXA)']
        datax = data[col[0]].str.contains('X')
        data_ = data[col[1]].str.contains('_')
        for i in range(0, len(data)):
            if datax[i] == True and data_[i] == True:
                data = data.drop(labels=i, axis=0)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def remove_nan(self, data, col_idx):
        cols = data.keys()
        col_list = list(cols)
        col = col_list[col_idx]
        data.dropna(subset=[col], inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def parseCT(self, _data):
        data = copy.deepcopy(_data)
        FIRST_CT_COL_IDX = 41
        COL_CNT = 52
        for i in range(FIRST_CT_COL_IDX, COL_CNT):
            data = self.remove_nan(data, i)
        return data
    
    def filterBlank(self, data, col):
        data[col].replace(' ', np.nan, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = data.dropna(subset=[col])
        data.reset_index(drop=True, inplace=True)
        for row in range(0, len(data.index)):
            if data.at[row, col] == '#DIV/0!':
                data.at[row, col] = 0
        return data
    
    def DeathBinary(self, _data, col):
        data = copy.deepcopy(_data)
        new_col = 'binary_'+col
        data[new_col] = 0 # alive
        loc = _data[col].loc[_data[col].notnull()]
        for row in loc:
            data.at[row, new_col] = 1 # dead
        data.reset_index(drop=True, inplace=True)
        return data
    
    def TableLookup(self, sex, age):
        return self.lookup_tab.at[sex, age]

    def Death(self, _data, col):
        data = copy.deepcopy(_data)
        new_col = '_'+col
        data[new_col] = data[col]
        loc = _data[_data[col].isnull()].index.tolist()
        for row in loc:
            if data.at[row, 'Sex'] == 0: ## female
                data.at[row, new_col] = self.lookup_tab.at[data.at[row, 'Age at CT'], 'Female'] * 365
            else:
                data.at[row, new_col] = self.lookup_tab.at[data.at[row, 'Age at CT'], 'Male'] * 365
#             if data.at[row, 'Age at CT'] > AVG_LIFE_EXPECTANCY:
#                 data.at[row, new_col] = data.at[row, 'Clinical F/U interval  [d from CT]'] + 85
#             else:
#                 data.at[row, new_col] = (AVG_LIFE_EXPECTANCY - data.at[row, 'Age at CT']) * 365
        data.reset_index(drop=True, inplace=True)
        return data
    
    def dropColumn(self, _data, col):
        data = _data.drop(col, axis=1)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def Encode(self):
        data = copy.deepcopy(self.data)
        data = self.Drop(data)
        data = self.parseCT(data)
        for col in data.columns:
            ## Clinical data
            if col == 'BMI':
                data[col] = self.BMI(data[col])
            elif col == 'BMI >30':
                data[col] = self.BMIrange(data[col])
            elif col == 'Sex':
                data[col] = self.Sex(data[col])
            elif col == 'Tobacco':
                data[col] = self.Tobacco(data[col])
            elif col == 'Alcohol abuse':
                data = self.Alcohol(data, col)
            elif col == 'FRS 10-year risk (%)':
                data[col] = self.FRS(data[col])
            elif col == 'FRAX 10y Fx Prob (Orange-w/ DXA)':
                data[col] = self.FRAX(data[col])
            elif col == 'FRAX 10y Hip Fx Prob (Orange-w/ DXA)':
                data[col] = self.FRAX(data[col])
            elif col == 'Met Sx':
                data[col] = self.MetSx(data[col])
            ## Clinical outcome
            elif col == 'DEATH [d from CT]':
                data = self.Death(data, col)
                data = self.DeathBinary(data, col)
            ## CT Data
            elif col == 'L1_HU_BMD' or col == 'TAT Area (cm2)' or col == 'Total Body                Area EA (cm2)' or col == 'VAT Area (cm2)' or col == 'SAT Area (cm2)' or col == 'VAT/SAT     Ratio' or col == 'Muscle HU' or col == ' Muscle Area (cm2)' or col == 'L3 SMI (cm2/m2)' or col == 'AoCa        Agatston' or col == 'Liver HU    (Median)':
                data = self.filterBlank(data, col)
#             elif col == 'VAT Area (cm2)' or col == 'Muscle Area (cm2)':
#                 data = self.dropColumn(data, col)
            else:
                print('unknown column: ', col)
        data = self.predictFRS(data)
        data = self.predictFRAX_Fx(data)
        data = self.predictFRAX_Hip(data)
        return data
