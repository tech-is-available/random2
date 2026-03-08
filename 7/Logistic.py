import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.datasets import load_iris

a = load_iris()
b = pd.DataFrame(a.data, columns=a.feature_names)

b['target'] = a.target
b = b[b['target'] != 2]

X = b.drop('target', axis=1)
Y = b['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Logistic Regression Metrics")
print("Accuracy: ", accuracy_score(Y_test,Y_pred))
print("Precision:", precision_score(Y_test,Y_pred))
print("Recall: ", recall_score(Y_test,Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test,Y_pred))
