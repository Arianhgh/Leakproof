import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

X = pd.read_csv("d.csv")
y = X.pop("target")
grid = GridSearchCV(SVC(random_state=0), {"C": [1, 10]})
nested = cross_val_score(grid, X, y)
print(nested.mean(), nested.std())
