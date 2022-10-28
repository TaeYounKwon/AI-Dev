import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine

wine = load_wine()

# print(wine.DESCR)

data = wine.data
label = wine.target
columns = wine.feature_names

data = pd.DataFrame(data, columns=columns)
data.head()

# k-Means

# 데이터 전처리
# MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# PCA (Reduce Dimention)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)

kmeans.fit(data)

cluster = kmeans.predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster,
            edgecolor='black',linewidth=1)

plt.savefig('my_figure.png')
plt.show()


from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,10))
children = single_clustering.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

dendrogram(linkage_matrix, p=len(data), labels=single_cluster,
           show_contracted=True, no_labels=True)