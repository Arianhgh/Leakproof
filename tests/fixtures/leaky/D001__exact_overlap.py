import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rows = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [5, 6, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0, 1]})
    train = rows.iloc[[0, 1, 2, 3, 4]].reset_index(drop=True)
    test = rows.iloc[[4, 5]].reset_index(drop=True)  # row 4 overlaps
    return DataAuditInput(train=train, test=test, target="target")
