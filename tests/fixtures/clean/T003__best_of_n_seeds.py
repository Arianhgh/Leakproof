import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scores = []
for seed in range(20):
    scores.append(cross_val_score(LogisticRegression(random_state=seed), X, y).mean())
print(np.mean(scores), np.std(scores))
