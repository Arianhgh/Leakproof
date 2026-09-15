import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("d.csv")
patient_id = df["patient_id"]
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for _train_idx, _test_idx in kf.split(df):
    pass
