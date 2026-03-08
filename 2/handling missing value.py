import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# ----- Load Dataset -----

a = pd.read_csv('handling.csv')

print("Original Data:\n", a)

# ----- Missing Value Handling (Age & Salary) -----

imputer = SimpleImputer(strategy='mean')

a[['Age', 'Salary']] = imputer.fit_transform(a[['Age', 'Salary']])

print("\nAfter Mean Imputation:\n", a)

# ----- Outlier Detection (Salary) -----

plt.boxplot(a['Salary'])
plt.title("Salary Outliers")
plt.show()

outliers = np.where(a['Salary'] > 80000)
print("\nOutliers in Salary:\n", outliers)

