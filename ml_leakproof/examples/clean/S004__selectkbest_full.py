import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
sel = SelectKBest(k=5)
X_train_s = sel.fit_transform(X_train, y_train)
X_test_s = sel.transform(X_test)
