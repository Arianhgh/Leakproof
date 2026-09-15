import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 80)
    train = pd.DataFrame({
        "noise": rng.rand(80),
        "leak": y,  # exact copy of the target
        "target": y,
    })
    test = train.iloc[:20].copy()
    return DataAuditInput(train=train, test=test, target="target")
