import pandas as pd
from urllib.request import urlopen
from pgmpy.models import BayesianModel


if __name__ == '__main__':

    data = pd.read_csv('data_cluster_handmade.csv')
    #print(data.dtypes)



    columns = ['L1_HU_BMD','Total Body                Area EA (cm2)','SAT Area (cm2)','VAT/SAT     Ratio','Muscle HU','L3 SMI (cm2/m2)','AoCa        Agatston','Liver HU    (Median)']
    y_columns = ['cluster']
    #death_day = ['_DEATH [d from CT]']

    _x = data[columns]
    _y = data[y_columns]
    print("hi there")

