import numpy as np
import pandas as pd

from ml_leakproof.data.input import DataAuditInput


def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "f1": rng.rand(200),
        "f2": rng.rand(200),
        "f3": rng.rand(200),
        "f4": rng.rand(200),
        "target": rng.randint(0, 2, 200),
    })
    test = pd.DataFrame({
        "f1": rng.rand(40) + 100.0,
        "f2": rng.rand(40) + 100.0,
        "f3": rng.rand(40) + 100.0,
        "f4": rng.rand(40) + 100.0,
        "target": rng.randint(0, 2, 40),
    })
    return DataAuditInput(train=train, test=test, target="target")
