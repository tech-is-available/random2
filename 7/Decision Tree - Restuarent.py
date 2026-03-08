import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

a = pd.read_csv("restuarant.csv")

a['Alt'] = a['Alt'].map({'No': 0, 'Yes': 1})
a['Bar'] = a['Bar'].map({'No': 0, 'Yes': 1})
a['Fri'] = a['Fri'].map({'No': 0, 'Yes': 1})
a['Hun'] = a['Hun'].map({'No': 0, 'Yes': 1})
a['Pat'] = a['Pat'].map({'None': 0, 'Some': 1, 'Full': 2})
a['Rain'] = a['Rain'].map({'No': 0, 'Yes': 1})
a['Res'] = a['Res'].map({'No': 0, 'Yes': 1})
a['Type'] = a['Type'].map({'French': 0, 'Thai': 1, 'Burger': 2, 'Italian': 3})
a['Est'] = a['Est'].map({'0-10': 0, '30-60': 1, '>60': 2})
a['Wait'] = a['Wait'].map({'No': 0, 'Yes': 1})


X = a.drop('Wait', axis=1)
Y = a['Wait']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = DecisionTreeClassifier().fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))




