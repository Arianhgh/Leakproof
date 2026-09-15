import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("d.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, test_idx in kf.split(df):
    pass
