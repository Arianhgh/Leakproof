import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
