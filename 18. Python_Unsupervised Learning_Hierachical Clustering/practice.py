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



# 데이터 전처리
# MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)






# Hierachical Clustering

from sklearn.cluster import AgglomerativeClustering

single_clustering = AgglomerativeClustering(n_clusters=3,linkage='single')
complete_clustering = AgglomerativeClustering(n_clusters=3,linkage='complete')
average_clustering = AgglomerativeClustering(n_clusters=3,linkage='average')

single_clustering.fit(data)
complete_clustering.fit(data)
average_clustering.fit(data)

single_cluster = single_clustering.labels_
complete_cluster = complete_clustering.labels_
average_cluster = average_clustering.labels_

print(single_cluster)
print(complete_cluster)
print(average_cluster)

plt.subplot(221)
plt.scatter(data[:,0],data[:,1],c=single_cluster)


plt.subplot(222)
plt.scatter(data[:,0],data[:,1],c=complete_cluster)

plt.subplot(223)
plt.scatter(data[:,0],data[:,1],c=average_cluster)

plt.subplot(224)
plt.scatter(data[:,0],data[:,1],c=label)

plt.savefig('clustering Figures.png')
# plt.show()


from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,10))
children = single_clustering.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

dendrogram(linkage_matrix, p=len(data), labels=single_cluster,show_contracted=True, no_labels=True)

# Silhouette Score

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
cluster = kmeans.predict(data)

best_n = 1
best_score = -1

for n_cluster in range(2,11):
  kmeans = KMeans(n_clusters=n_cluster)
  kmeans.fit(data)
  cluster = kmeans.predict(data)
  score = silhouette_score(data, cluster)

  print('Number of Cluster: {} Silhoutette Score:{:.2f}'.format(n_cluster, score))

  if score > best_score:
    best_n = n_cluster
    best_score = score

print('Number of Cluster that has Max Sillhoutette Score: {}\n Silhoutette Score {:.2f}'.format(best_n, best_score))
print('\n')
print('New Silhouette Test')

for n_cluster in range(2,11):
  average_clustering = AgglomerativeClustering(n_clusters=n_cluster,linkage='average')
  average_clustering.fit(data)
  cluster = average_clustering.labels_
  score = silhouette_score(data, cluster)

  print('Number of Cluster: {} Silhoutette Score:{:.2f}'.format(n_cluster, score))

  if score > best_score:
    best_n = n_cluster
    best_score = score

print('Number of Cluster that has Max Sillhoutette Score: {}\n Silhoutette Score {:.2f}'.format(best_n, best_score))