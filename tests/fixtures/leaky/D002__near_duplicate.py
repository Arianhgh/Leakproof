import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    base = rng.rand(30, 4)
    train = pd.DataFrame(base, columns=list("abcd"))
    train["target"] = (base[:, 0] > 0.5).astype(int)
    test_base = base[:10] + 1e-6  # near-identical to train rows
    test = pd.DataFrame(test_base, columns=list("abcd"))
    test["target"] = (test_base[:, 0] > 0.5).astype(int)
    return DataAuditInput(train=train, test=test, target="target")
