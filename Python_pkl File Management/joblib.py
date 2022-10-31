from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib

dataset = datasets.load_iris()

X, y = dataset['data'], dataset['target']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

joblib.dump(model, './knn_model.pkl')

loaded_model = joblib.load('./knn_model.pkl')

score = loaded_model.score(X, y)
print('Accuracy: {score:.3f}'.format(score=score))