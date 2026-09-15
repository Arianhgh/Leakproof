import pandas as pd
from sklearn.model_selection import GroupKFold

df = pd.read_csv("d.csv")
groups = df["patient_id"]
gkf = GroupKFold(n_splits=5)
for _train_idx, _test_idx in gkf.split(df, groups=groups):
    pass
