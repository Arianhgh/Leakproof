import numpy as np

def run():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = rng.randint(0, 2, 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    m = LogisticRegression(random_state=0).fit(X_train, y_train)
    m.score(X_test, y_test)  # scored once
