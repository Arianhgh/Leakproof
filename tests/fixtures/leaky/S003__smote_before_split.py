import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=0)
