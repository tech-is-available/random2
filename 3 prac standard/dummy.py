import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

a = pd.read_csv("dummy.csv")
print("Original DataFrame:\n", a)

b = pd.get_dummies(a)

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

print("\nDataFrame after Feature Dummification:\n", b)

X = b.drop("Purchased_Yes", axis=1)
Y = b["Purchased_Yes"]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
