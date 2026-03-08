import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset("tips")

# Scatter plot
plt.figure(figsize=(10,6))
plt.scatter(data['total_bill'], data['tip'], alpha=0.5)
plt.title('Relationship between Total Bill and Tip', fontsize=16)
plt.xlabel('Total Bill', fontsize=14)
plt.ylabel('Tip', fontsize=14)
plt.show()

# Bar chart
plt.figure(figsize=(10,6))
sns.countplot(x='day', data=data)
plt.title('Distribution of Days', fontsize=16)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
numerical_cols = ['total_bill','tip','size']
sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

print("Title: Restaurant Sales Analysis")

print("\nScatter Plot: Relationship between Total Bill and Tip")
print("Figure 1: Scatter Plot of Total Bill vs Tip")

print("\nBar Chart: Distribution of customers by day")
print("Figure 2: Distribution of Days")

print("\nHeatmap: Correlation between Total Bill, Tip and Group Size")
print("Figure 3: Correlation Heatmap")

print("\nThese visualizations help understand customer spending patterns.")
