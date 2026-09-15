import numpy as np
import pandas as pd

from ml_leakproof.data.input import DataAuditInput


def make_input():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    train = pd.DataFrame({
        "f1": rng.rand(200),
        "f2": rng.rand(200),
        "f3": rng.rand(200),
        "proxy": y + rng.normal(0, 0.01, 200),  # near-deterministic proxy
        "target": y,
    })
    test = train.iloc[:40].copy()
    return DataAuditInput(train=train, test=test, target="target")
