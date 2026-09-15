import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
imp = SimpleImputer(strategy="mean")
X_train_i = imp.fit_transform(X_train)
X_test_i = imp.transform(X_test)
