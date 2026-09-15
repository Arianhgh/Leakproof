import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "f1": rng.rand(80),
        "f2": rng.rand(80),
        "target": rng.randint(0, 2, 80),
    })
    # test rows are well separated from train (offset by +100) so there is no
    # incidental exact/near-duplicate overlap to muddy this clean example.
    test = pd.DataFrame({
        "f1": rng.rand(20) + 100.0,
        "f2": rng.rand(20) + 100.0,
        "target": rng.randint(0, 2, 20),
    })
    return DataAuditInput(train=train, test=test, target="target")
