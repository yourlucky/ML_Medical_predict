import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt



_data = pd.read_csv('data_bioage.csv') 
columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
y_columns = ['I_age']

#_data.plot(kind = 'scatter', y='SAT Area (cm2)',x= 'I_age')
_data=_data[_data['_death_in_3yr']==0]
valid_data=_data.groupby(['I_age'],as_index=False).mean()
valid_data.plot(kind = 'line', y='SAT Area (cm2)',x= 'I_age')
_data.plot(kind = 'scatter', y='SAT Area (cm2)',x= 'I_age')
#plt.show()

print(valid_data['SAT Area (cm2)'])