import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("d.csv")
y = df.pop("target")
train, test, y_train, y_test = train_test_split(df, y, random_state=0)
scaler = StandardScaler()
scaler.fit(train)
