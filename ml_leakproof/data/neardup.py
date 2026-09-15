"""Bounded near-duplicate candidate generation for supported modalities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


class NearDuplicatePairs(list[tuple[int, int, float]]):
    """List-compatible result carrying truthful resource coverage."""

    def __init__(
        self,
        values: Iterable[tuple[int, int, float]] = (),
        *,
        complete: bool = True,
        sampled: bool = False,
        unavailable: str | None = None,
        candidates_checked: int = 0,
        comparisons: int = 0,
        unreadable: int = 0,
    ):
        super().__init__(values)
        self.complete = complete
        self.sampled = sampled
        self.unavailable = unavailable
        self.candidates_checked = candidates_checked
        self.comparisons = comparisons
        self.unreadable = unreadable


def _shingles(text: str, k: int = 5) -> set[str]:
    normalized = " ".join(str(text).lower().split())
    if len(normalized) <= k:
        return {normalized} if normalized else set()
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def text_near_duplicates(
    train_texts: Iterable[Any],
    test_texts: Iterable[Any],
    threshold: float,
    *,
    max_candidates: int = 100_000,
    max_pairs: int = 100_000,
) -> NearDuplicatePairs:
    """Find similar text rows with a bounded inverted-shingle index.

    The optional datasketch dependency used to return an unbounded candidate
    list from ``MinHashLSH.query`` before the cap could be checked. The
    deterministic inverted index below keeps both candidate storage and pair
    comparisons bounded, and reports ``complete=False`` when a limit cuts the
    scan short.
    """

    if max_candidates <= 0 or max_pairs <= 0:
        raise ValueError("near-duplicate limits must be positive")
    train_texts = list(train_texts)
    test_texts = list(test_texts)
    return _bounded_jaccard(
        train_texts,
        test_texts,
        threshold,
        max_candidates=max_candidates,
        max_pairs=max_pairs,
    )


def _bounded_jaccard(
    train_texts: Iterable[Any],
    test_texts: Iterable[Any],
    threshold: float,
    *,
    max_candidates: int,
    max_pairs: int,
) -> NearDuplicatePairs:
    train_texts = list(train_texts)
    test_texts = list(test_texts)
    train_shingles = [_shingles(value) for value in train_texts]
    postings: dict[str, list[int]] = defaultdict(list)
    truncated_postings: set[str] = set()
    for index, shingles in enumerate(train_shingles):
        for shingle in shingles:
            posting = postings[shingle]
            if len(posting) < max_candidates:
                posting.append(index)
            else:
                truncated_postings.add(shingle)
    pairs: list[tuple[int, int, float]] = []
    candidates = 0
    complete = True
    for test_index, text in enumerate(test_texts):
        test_shingles = _shingles(text)
        candidate_ids: list[int] = []
        seen_candidates: set[int] = set()
        for shingle in sorted(test_shingles):
            if shingle in truncated_postings:
                # A frequent shingle had more candidates than the configured
                # storage bound; do not claim a complete negative scan.
                complete = False
            for index in postings.get(shingle, []):
                if index in seen_candidates:
                    continue
                seen_candidates.add(index)
                candidate_ids.append(index)
                if len(candidate_ids) > max_candidates:
                    complete = False
                    break
            if not complete:
                break
        for train_index in candidate_ids:
            candidates += 1
            if candidates > max_candidates or len(pairs) >= max_pairs:
                complete = False
                break
            train_set = train_shingles[train_index]
            if not train_set or not test_shingles:
                continue
            score = len(train_set & test_shingles) / len(train_set | test_shingles)
            if score >= threshold:
                pairs.append((train_index, test_index, float(score)))
        if not complete:
            break
    return NearDuplicatePairs(
        pairs,
        complete=complete,
        candidates_checked=candidates,
        comparisons=candidates,
    )


def numeric_near_duplicates(
    train_X: Any,
    test_X: Any,
    distance: float,
    *,
    max_candidates: int = 100_000,
    max_pairs: int = 100_000,
) -> NearDuplicatePairs:
    if max_candidates <= 0 or max_pairs <= 0:
        raise ValueError("near-duplicate limits must be positive")
    sampled = False
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return NearDuplicatePairs(complete=False, unavailable=str(exc))
    if len(train_X) == 0 or len(test_X) == 0:
        return NearDuplicatePairs()
    try:
        train_array = np.asarray(train_X, dtype=float)
        test_array = np.asarray(test_X, dtype=float)
        if len(test_array) > max_candidates:
            test_array = test_array[:max_candidates]
            sampled = True
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_array)
        test_scaled = scaler.transform(test_array)
    except Exception as exc:
        return NearDuplicatePairs(complete=False, unavailable=f"numeric preprocessing unavailable: {exc}")
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(train_scaled)
    distances, indices = neighbors.kneighbors(test_scaled)
    pairs: list[tuple[int, int, float]] = []
    for test_index, (distance_value, train_index) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if len(pairs) >= max_pairs:
            return NearDuplicatePairs(
                pairs,
                complete=False,
                sampled=sampled,
                candidates_checked=test_index,
                comparisons=test_index,
            )
        norm = float(distance_value) / ((train_scaled.shape[1] ** 0.5) or 1.0)
        if norm <= distance:
            pairs.append((int(train_index), test_index, norm))
    return NearDuplicatePairs(
        pairs,
        complete=not sampled,
        sampled=sampled,
        candidates_checked=len(test_array),
        comparisons=len(test_array),
    )


def image_near_duplicates(
    train_paths: Iterable[str],
    test_paths: Iterable[str],
    max_distance: int,
    *,
    max_candidates: int = 100_000,
    max_pairs: int = 100_000,
) -> NearDuplicatePairs:
    if max_candidates <= 0 or max_pairs <= 0:
        raise ValueError("near-duplicate limits must be positive")
    train_paths = list(train_paths)
    test_paths = list(test_paths)
    try:
        from PIL import Image
    except Exception as exc:
        return NearDuplicatePairs(complete=False, unavailable=f"image backend unavailable: {exc}")
    try:
        import imagehash
    except Exception:
        imagehash = None

    unreadable = 0

    def hash_one(path: str) -> Any:
        nonlocal unreadable
        try:
            with Image.open(path) as image:
                if imagehash is not None:
                    return imagehash.phash(image)
                # A dependency-free perceptual fallback keeps image checks
                # available with Pillow-only data installs.  It deliberately
                # reports a perceptual distance rather than pretending to be
                # cryptographic equality.
                gray = image.convert("L").resize((8, 8))
                pixels = list(
                    gray.get_flattened_data()
                    if hasattr(gray, "get_flattened_data")
                    else gray.getdata()
                )
                mean = sum(pixels) / max(1, len(pixels))
                return sum((1 << index) for index, value in enumerate(pixels) if value >= mean)
        except Exception:
            unreadable += 1
            return None

    train_limited = len(train_paths) > max_candidates
    test_limited = len(test_paths) > max_candidates
    train_paths = train_paths[:max_candidates]
    test_paths = test_paths[:max_candidates]
    train_hashes = [(index, hash_one(path)) for index, path in enumerate(train_paths)]
    test_hashes = [(index, hash_one(path)) for index, path in enumerate(test_paths)]
    train_hashes = [(index, value) for index, value in train_hashes if value is not None]
    test_hashes = [(index, value) for index, value in test_hashes if value is not None]
    pairs: list[tuple[int, int, float]] = []
    comparisons = 0
    for test_index, test_hash in test_hashes:
        best: tuple[int, int] | None = None
        for train_index, train_hash in train_hashes:
            comparisons += 1
            if comparisons > max_candidates or len(pairs) >= max_pairs:
                return NearDuplicatePairs(
                    pairs,
                    complete=False,
                    sampled=train_limited or test_limited,
                    candidates_checked=comparisons,
                    comparisons=comparisons,
                    unreadable=unreadable,
                )
            if imagehash is not None:
                distance = int(test_hash - train_hash)
            else:
                distance = (int(test_hash) ^ int(train_hash)).bit_count()
            if best is None or distance < best[1]:
                best = (train_index, distance)
        if best is not None and best[1] <= max_distance:
            pairs.append((best[0], test_index, float(best[1])))
    return NearDuplicatePairs(
        pairs,
        complete=not (train_limited or test_limited),
        sampled=train_limited or test_limited,
        candidates_checked=comparisons,
        comparisons=comparisons,
        unreadable=unreadable,
    )
