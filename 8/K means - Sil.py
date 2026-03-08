
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

a = load_iris()
b = pd.DataFrame(a.data, columns=a.feature_names)

X = StandardScaler().fit_transform(b)

scores = []
for i in range(2,11):
    model = KMeans(n_clusters=i)
    labels = model.fit_predict(X)
    scores.append(silhouette_score(X, labels))


plt.plot(range(2,11), scores, marker='o')
plt.title("Silhouette Method")
plt.show()


model = KMeans(n_clusters=3)
labels = model.fit_predict(X)


X2 = PCA(n_components=2).fit_transform(X)


plt.scatter(X2[:,0], X2[:,1], c=labels)
plt.title("K-Means Clusters")
plt.show()


b['cluster'] = labels
for i, group in b.groupby('cluster'):
    print(f"\n--- Cluster {i} ---")
    print(group.describe())

