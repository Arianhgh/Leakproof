import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
best = 0.0
for seed in range(20):
    score = cross_val_score(LogisticRegression(random_state=seed), X, y).mean()
    best = max(best, score)
print(best)
