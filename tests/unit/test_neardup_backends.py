"""Optional backends, real images, and bounded similarity work."""

import sys

import numpy as np
import pytest

from ml_leakproof.data.neardup import (
    image_near_duplicates,
    numeric_near_duplicates,
    text_near_duplicates,
)


@pytest.mark.parametrize(
    "backend,args",
    [
        (text_near_duplicates, (["x"], ["x"], 0.9)),
        (numeric_near_duplicates, ([[0]], [[0]], 0.1)),
        (image_near_duplicates, ([], [], 1)),
    ],
)
def test_invalid_work_limits(backend, args):
    with pytest.raises(ValueError):
        backend(*args, max_candidates=0)
    with pytest.raises(ValueError):
        backend(*args, max_pairs=0)


def test_text_jaccard_and_global_bounds():
    result = text_near_duplicates(["hello world", "unrelated"], ["hello world", ""], 0.9)
    assert result == [(0, 0, 1.0)] and result.complete
    capped = text_near_duplicates(["hello world"] * 10, ["hello world"], 0.9, max_candidates=2)
    assert not capped.complete and len(capped) <= 2
    pairs = text_near_duplicates(["hello world"] * 3, ["hello world"], 0.9, max_pairs=1)
    assert len(pairs) == 1 and not pairs.complete
    assert not text_near_duplicates([""], [""], 0.9)


def test_numeric_success_sampling_invalid_and_empty():
    assert numeric_near_duplicates([], [], 0.1).complete
    train = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert numeric_near_duplicates(train, train[:1], 0.01) == [(0, 0, 0.0)]
    sampled = numeric_near_duplicates(train, train, 0.01, max_candidates=1)
    assert sampled.sampled and not sampled.complete
    capped = numeric_near_duplicates(train, train, 0.01, max_pairs=1)
    assert len(capped) == 1 and not capped.complete
    invalid = numeric_near_duplicates([["no"]], [[1]], 0.1)
    assert invalid.unavailable and not invalid.complete


def test_numeric_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", None)
    result = numeric_near_duplicates([[0]], [[0]], 0.1)
    assert result.unavailable and not result.complete


@pytest.mark.parametrize("fallback", [False, True])
def test_real_images_and_unreadable_inputs(tmp_path, monkeypatch, fallback):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(path)
    if fallback:
        monkeypatch.setitem(sys.modules, "imagehash", None)
    result = image_near_duplicates([str(path)], [str(path)], 1)
    assert result == [(0, 0, 0.0)] and result.complete
    unreadable = image_near_duplicates([str(path), str(tmp_path / "missing")], [str(path)], 1)
    assert unreadable.unreadable == 1 and not unreadable.complete
    capped = image_near_duplicates([str(path)] * 2, [str(path)] * 2, 1, max_candidates=1)
    assert capped.sampled and not capped.complete
    pairs = image_near_duplicates([str(path)], [str(path)] * 2, 1, max_pairs=1)
    assert len(pairs) == 1 and not pairs.complete


def test_image_backend_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    result = image_near_duplicates([], [], 1)
    assert result.unavailable and not result.complete
