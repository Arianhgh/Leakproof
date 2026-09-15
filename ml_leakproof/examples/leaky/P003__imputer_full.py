import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
imp = SimpleImputer(strategy="mean")
X_imp = imp.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imp, y, random_state=0)
