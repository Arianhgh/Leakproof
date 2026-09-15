import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("d.csv")
y = X.pop("target")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y)
