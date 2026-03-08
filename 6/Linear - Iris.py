import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

a = load_iris()
b = pd.DataFrame(a.data,columns=a.feature_names)
print(b)

X = b[['sepal length (cm)']]
Y = b['sepal width (cm)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R-squared:", r2_score(Y_test, Y_pred))
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
