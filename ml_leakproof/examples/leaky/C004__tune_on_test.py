import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X_test, y_test)
