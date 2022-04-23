import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class K_mean_add_csv:
    def __init__(self, _path,cluster_number=7):
        self._data = pd.read_csv(_path)
        self.cluster_number=cluster_number
        #self.x = self._data[['_DEATH [d from CT]']]
        #self._data.sort_values(by=['_DEATH [d from CT]'])
        self.x = self._data[['Age at CT']]
        #self._data.sort_values(by=['Age at CT'])
        self.y = pd.DataFrame(self.K_mean_clustering())
        self._data['cluster_3']= self.y     

    def K_mean_clustering(self) :
        _kmean =  KMeans(n_clusters=self.cluster_number)
        _kmean.fit(self.x)
        result_kmeans = self.x.copy()
        result_kmeans['cluster_3'] = _kmean.labels_
        print(result_kmeans)
        
        return result_kmeans['cluster_3']
    

if __name__ == '__main__':
    #preprocessor = Preprocessor('OppScrData.csv')
    #data = preprocessor.Encode()
    #data.to_csv('data.csv', index=True)
    K_mean_add = K_mean_add_csv('data.csv')
    K_mean_add._data.to_csv('data_cluster_age.csv',index=True)

    data = pd.read_csv('data_cluster.csv')
    _x =data['cluster']
    _y =data['_DEATH [d from CT]']
    #_y =data['Age at CT']


    plt.scatter(_x,_y)
    plt.show()


#     display(data)


