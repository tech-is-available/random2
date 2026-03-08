import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


a = pd.read_csv("standard.csv")
b = a[['Age', 'Salary']]

X_std = StandardScaler().fit_transform(b)
X_norm = MinMaxScaler().fit_transform(b)

plt.subplot(1,3,1)
plt.scatter(b.iloc[:,0], b.iloc[:,1])
plt.title("Original")

plt.subplot(1,3,2)
plt.scatter(X_std[:,0], X_std[:,1])
plt.title("Standardized")

plt.subplot(1,3,3)
plt.scatter(X_norm[:,0], X_norm[:,1])
plt.title("Normalized")

plt.show()
