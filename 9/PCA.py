import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

a = load_iris()
b = pd.DataFrame(a.data,columns=a.feature_names)

y = a.target
X = StandardScaler().fit_transform(b)

pca = PCA()
pca.fit(X)

# Explained variance plot
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance Ratio")
plt.show()

# Find components for 95%
n = np.argmax(cum_var >= 0.95) + 1
print("Components for 95% variance:", n)

# Reduce dimensions
X_reduced = PCA(n_components=n).fit_transform(X)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Reduced Data")
plt.show()

