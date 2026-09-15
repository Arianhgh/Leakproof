import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
print(accuracy_score(y_train, m.predict(X_train)))
