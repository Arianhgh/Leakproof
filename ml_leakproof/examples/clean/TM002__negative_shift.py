import pandas as pd

df = pd.read_csv("d.csv")
df["lag_target"] = df["target"].shift(1)
