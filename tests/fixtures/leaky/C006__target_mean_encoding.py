import pandas as pd

df = pd.read_csv("d.csv")
df["cat_enc"] = df.groupby("cat")["target"].transform("mean")
