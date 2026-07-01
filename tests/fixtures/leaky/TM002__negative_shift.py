import pandas as pd

df = pd.read_csv("d.csv")
df["future_target"] = df["target"].shift(-1)
