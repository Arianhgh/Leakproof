import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6], "pid": ["p1", "p1", "p2", "p2", "p3", "p3"], "target": [0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [0.7, 0.8], "pid": ["p4", "p5"], "target": [0, 1]})
    return DataAuditInput(train=train, test=test, target="target", group="pid")
