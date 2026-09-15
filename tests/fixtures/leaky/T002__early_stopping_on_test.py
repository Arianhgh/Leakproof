import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = XGBClassifier(early_stopping_rounds=10, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
