import pandas as pd
from leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"f": list(range(100)), "target": [0] * 97 + [1] * 3})
    test = pd.DataFrame({"f": list(range(20)), "target": [0] * 19 + [1]})
    return DataAuditInput(train=train, test=test, target="target")
