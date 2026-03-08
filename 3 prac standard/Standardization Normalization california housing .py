import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

a = fetch_california_housing()
b = pd.DataFrame(a.data, columns=a.feature_names)

# Take only a few rows to avoid congestion
c = b.iloc[:300]

X_std = StandardScaler().fit_transform(c)
X_norm = MinMaxScaler().fit_transform(c)


model = KMeans(n_clusters=3)
y = model.fit_predict(c)


plt.subplot(1,3,1)
plt.scatter(c.iloc[:,0], c.iloc[:,1],c=y)
plt.title("Original")

plt.subplot(1,3,2)
plt.scatter(X_std[:,0], X_std[:,1],c=y)
plt.title("Standardized")

plt.subplot(1,3,3)
plt.scatter(X_norm[:,0], X_norm[:,1],c=y)
plt.title("Normalized")

plt.show()
