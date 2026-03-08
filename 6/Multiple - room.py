import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

a = pd.read_csv("room.csv")
print(a)

X = a.drop('Price', axis=1)
Y = a['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("\n----- Multiple Linear Regression -----")
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R-squared:", r2_score(Y_test, Y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
