import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scores = cross_val_score(LogisticRegression(random_state=0), X, y)
print(scores.mean())
