import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
train, test = train_test_split(df, random_state=0)
df["cat_enc"] = df.groupby("cat")["target"].transform("mean")
