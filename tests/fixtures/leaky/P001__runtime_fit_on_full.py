import numpy as np

def run():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = rng.randint(0, 2, 60)
    scaler = StandardScaler()
    scaler.fit_transform(X)  # fit on full data before split -> sees the test rows
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
