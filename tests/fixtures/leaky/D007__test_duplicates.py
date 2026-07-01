import pandas as pd
from leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    test = pd.DataFrame({"a": [4, 4, 4, 5], "target": [1, 1, 1, 0]})  # dup rows
    return DataAuditInput(train=train, test=test, target="target")
