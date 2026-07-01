"""Near-duplicate detection across splits.

Strategy by data type:
  - text columns       -> MinHash + LSH (datasketch) on shingles
  - numeric features   -> NearestNeighbors in standardized space
  - image path columns -> perceptual hash (imagehash) [opt-in via feature_types]
"""

from __future__ import annotations

from typing import Any


def _shingles(text: str, k: int = 5) -> set[str]:
    text = " ".join(str(text).lower().split())
    if len(text) <= k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def text_near_duplicates(train_texts, test_texts, threshold: float) -> list[tuple[int, int, float]]:
    """Return (train_idx, test_idx, jaccard) pairs above ``threshold``.

    Uses datasketch MinHashLSH when available; falls back to exact Jaccard for
    small inputs.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except Exception:
        return _exact_jaccard(train_texts, test_texts, threshold)

    num_perm = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    train_mh: dict[int, Any] = {}
    for i, t in enumerate(train_texts):
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(t):
            m.update(sh.encode("utf-8"))
        train_mh[i] = m
        lsh.insert(f"train-{i}", m)

    pairs: list[tuple[int, int, float]] = []
    for j, t in enumerate(test_texts):
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(t):
            m.update(sh.encode("utf-8"))
        for key in lsh.query(m):
            i = int(key.split("-", 1)[1])
            est = train_mh[i].jaccard(m)
            if est >= threshold:
                pairs.append((i, j, float(est)))
    return pairs


def _exact_jaccard(train_texts, test_texts, threshold: float) -> list[tuple[int, int, float]]:
    train_sh = [(_shingles(t)) for t in train_texts]
    pairs: list[tuple[int, int, float]] = []
    for j, t in enumerate(test_texts):
        sj = _shingles(t)
        if not sj:
            continue
        for i, si in enumerate(train_sh):
            if not si:
                continue
            inter = len(si & sj)
            if inter == 0:
                continue
            jac = inter / len(si | sj)
            if jac >= threshold:
                pairs.append((i, j, jac))
    return pairs


def numeric_near_duplicates(train_X, test_X, distance: float) -> list[tuple[int, int, float]]:
    """Return (train_idx, test_idx, dist) pairs within ``distance`` in standardized space."""
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return []

    if len(train_X) == 0 or len(test_X) == 0:
        return []

    scaler = StandardScaler()
    tr = scaler.fit_transform(np.asarray(train_X, dtype=float))
    te = scaler.transform(np.asarray(test_X, dtype=float))
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(tr)
    dists, idxs = nn.kneighbors(te)
    pairs: list[tuple[int, int, float]] = []
    # normalize distance by sqrt(n_features) so the threshold is dimension-stable
    denom = (tr.shape[1] ** 0.5) or 1.0
    for j, (d, i) in enumerate(zip(dists[:, 0], idxs[:, 0])):
        norm = float(d) / denom
        if norm <= distance:
            pairs.append((int(i), j, norm))
    return pairs
