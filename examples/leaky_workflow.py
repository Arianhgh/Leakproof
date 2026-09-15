"""An intentionally unsafe workflow for demonstrating findings."""

import numpy as np


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    X = rng.rand(120, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.75).astype(int)

    # The scaler learns from all rows before the holdout is created.
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y,
    )
    scores = cross_val_score(
        LogisticRegression(max_iter=1000, random_state=7),
        X_scaled,
        y,
        cv=3,
    )
    print(f"leaky CV accuracy: {scores.mean():.3f}; holdout rows: {len(X_test)}")
    print(f"training rows: {len(X_train)}; held-out labels: {len(y_test)}")


if __name__ == "__main__":
    main()
