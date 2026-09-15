import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    import numpy as np

    rng = np.random.RandomState(0)
    # balanced target, uncorrelated with the feature (no leakage to find)
    train = pd.DataFrame({"f": rng.rand(100), "target": rng.randint(0, 2, 100)})
    test = pd.DataFrame({"f": rng.rand(20) + 100.0, "target": rng.randint(0, 2, 20)})
    return DataAuditInput(train=train, test=test, target="target")
