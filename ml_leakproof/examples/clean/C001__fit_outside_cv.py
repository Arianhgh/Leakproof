import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("d.csv")
y = X.pop("target")
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
scores = cross_val_score(pipe, X, y)
