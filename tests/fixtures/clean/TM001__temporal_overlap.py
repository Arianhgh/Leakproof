import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.05], "ts": pd.date_range("2020-01-01", periods=10), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [100.0, 101.0], "ts": pd.to_datetime(["2021-01-01", "2021-02-01"]), "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target", time="ts")
