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

class InputParser:
    def __init__(self, path):
        self.drop_num = 38
        self.data = self.ReadInput(path)
    
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
        avg = total / cnt
        datax = data.str.contains('X')
        for i in range(0, len(data)):
            if datax[i] == True:
                data[i] = avg
        return data
    
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
                data[i] = avg
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
    
    def Encode(self):
        data = copy.deepcopy(self.data)
        data = self.Drop(data)
        data = self.parseCT(data)
        for col in data.columns:
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
#             else:
#                 print('unknown column: ', col)

        return data

if __name__ == '__main__':
    parser = InputParser('OppScrData.csv')
    data = parser.Encode()
    data.to_csv('data.csv', index=True)
    display(data)


