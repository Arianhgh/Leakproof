import pandas as pd

from ml_leakproof.data.input import DataAuditInput


def make_input():
    train = pd.DataFrame({"a": [1, 2, 3], "b": [5, 6, 7], "target": [0, 1, 0]})
    test = pd.DataFrame({"a": [4, 5], "b": [8, 9], "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target")
