import pandas as pd

df = pd.read_csv("d.csv")
df["future_feature"] = df["value"].shift(-1)
