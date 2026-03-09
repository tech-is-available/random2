import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from scipy import stats

a=pd.read_csv('missing.csv')
print(a)

imputer=SimpleImputer(strategy='mean')
a[['Age','Salary']]=imputer.fit_transform(a[['Age','Salary']])
print(a)

lower = 40000
upper = 80000

plt.boxplot(a['Salary'], flierprops=dict(marker='o', markersize=8))
plt.axhline(lower, linestyle='--')   # show lower limit
plt.axhline(upper, linestyle='--')   # show upper limit
plt.show()

outliers = np.where((a['Salary'] < lower) | (a['Salary'] > upper))
print(outliers)

