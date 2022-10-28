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
