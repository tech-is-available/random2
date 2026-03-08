
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
# ----- Load Dataset -----
a = load_diabetes()
b = pd.DataFrame(a.data, columns=a.feature_names)
print("Original Data:")
print(b.head())
# ----- Outlier Detection (BMI feature) -----
plt.boxplot(b['bmi'])
plt.title("BMI Outliers")
plt.show()
# Detect outliers using a threshold (example: values > 0.05)
outliers = np.where(b['bmi'] > 0.05)
print("Outliers in BMI:")
print(outliers)
