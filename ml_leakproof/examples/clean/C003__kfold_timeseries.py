import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("d.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
tss = TimeSeriesSplit(n_splits=5)
for _train_idx, _test_idx in tss.split(df):
    pass
