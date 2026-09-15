import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("d.csv")
y = df.pop("target")
pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(random_state=0))
