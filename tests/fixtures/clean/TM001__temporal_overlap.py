import pandas as pd
from leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [1, 2, 3], "ts": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]), "target": [0, 1, 0]})
    test = pd.DataFrame({"x": [4, 5], "ts": pd.to_datetime(["2021-01-01", "2021-02-01"]), "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target", time="ts")
