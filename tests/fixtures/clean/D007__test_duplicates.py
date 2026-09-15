import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [5, 6, 7, 8, 9], "target": [1, 0, 1, 0, 1]})
    return DataAuditInput(train=train, test=test, target="target")
