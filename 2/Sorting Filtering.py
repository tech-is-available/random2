import pandas as pd

a = pd.read_csv('sorting.csv')
print("Original Data:\n", a)


# -------- Display cars with Sell Price > 4000 --------
filtered_data = a[a['Sell Price'] > 4000]
print("\nCars with Sell Price > 4000:\n", filtered_data)


# -------- Sort data in ascending order (by Sell Price) --------
sorted_data = a.sort_values(by='Sell Price')
print("\nSorted Data (Ascending by Sell Price):\n", sorted_data)


# -------- Group data by Make --------
grouped_data = a.groupby('Make')
print("\nGrouped Data (First record of each Make):\n", grouped_data.first())

for Make, data in a.groupby('Make'):
    print(f"\n{Make}")
    print(data)

print(a.sort_values('Make'))
