import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

a = pd.read_csv("windy.csv")

a['Outlook'] = a['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
a['Wind'] = a['Wind'].map({'Weak': 0, 'Strong': 1})
a['PlayTennis'] = a['PlayTennis'].map({'No': 0, 'Yes': 1})


X = a.drop('PlayTennis', axis=1)
Y = a['PlayTennis']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier().fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
