import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
sel = SelectKBest(k=5)
X_sel = sel.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=0)
