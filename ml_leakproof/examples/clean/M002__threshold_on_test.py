import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_tr, X_test, y_tr, y_test = train_test_split(X, y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
probs = m.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, probs)
