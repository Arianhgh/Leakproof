import pandas as pd

from ml_leakproof.data.input import DataAuditInput


def make_input():
    train = pd.DataFrame({"x": [1, 2, 3], "pid": ["p1", "p2", "p3"], "target": [0, 1, 0]})
    test = pd.DataFrame({"x": [4, 5], "pid": ["p4", "p5"], "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target", group="pid")
