"""A clean train/test workflow for the ml-leakproof runtime checker."""

import numpy as np


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    X = rng.rand(120, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.75).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=7).fit(
        X_train_scaled, y_train
    )
    print(f"clean test accuracy: {model.score(X_test_scaled, y_test):.3f}")


if __name__ == "__main__":
    main()
