import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
y = df.pop("target")
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)
