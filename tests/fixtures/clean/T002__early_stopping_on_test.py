import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_tr, X_test, y_tr, y_test = train_test_split(X, y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state=0)
model = XGBClassifier(early_stopping_rounds=10, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
