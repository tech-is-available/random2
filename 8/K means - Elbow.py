import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


a = load_iris()
b = pd.DataFrame(a.data, columns=a.feature_names)

X = StandardScaler().fit_transform(b)


#ELBOW
wcss = []
for i in range(1,11):
    model = KMeans(n_clusters=i)
    model.fit(X)
    wcss.append(model.inertia_)
plt.plot(wcss, marker='o')
plt.title("Elbow Method")
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
