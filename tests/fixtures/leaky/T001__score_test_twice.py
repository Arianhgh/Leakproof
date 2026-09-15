import numpy as np

def run():
    # imports happen inside run() so they resolve to the instrumented functions,
    # mirroring how `ml_leakproof run script.py` executes user code under the hooks.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = rng.randint(0, 2, 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    m = LogisticRegression(random_state=0).fit(X_train, y_train)
    m.score(X_test, y_test)
    m.score(X_test, y_test)  # scored twice -> T001
