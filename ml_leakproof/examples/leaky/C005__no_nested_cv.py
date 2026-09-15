import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X = pd.read_csv("d.csv")
y = X.pop("target")
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X, y)
print(grid.best_score_)
