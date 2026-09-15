import numpy as np
import pandas as pd

from ml_leakproof.data.input import DataAuditInput


def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame(rng.rand(30, 4), columns=list("abcd"))
    train["target"] = rng.randint(0, 2, 30)
    test = pd.DataFrame(rng.rand(10, 4) + 10.0, columns=list("abcd"))
    test["target"] = rng.randint(0, 2, 10)
    return DataAuditInput(train=train, test=test, target="target")
