import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X = pd.read_csv("d.csv")
y = X.pop("target")
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X, y)
print(grid.best_score_)
