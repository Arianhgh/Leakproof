"""Paired-fixture tests: every rule must flag its leaky example and stay silent
on its clean example, and the clean set must be entirely false-positive free."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._harness import CLEAN, LEAKY, rule_id_of, run_fixture

LEAKY_FILES = sorted(LEAKY.glob("*.py"))
CLEAN_FILES = sorted(CLEAN.glob("*.py"))


@pytest.mark.parametrize("path", LEAKY_FILES, ids=lambda p: p.name)
def test_leaky_fixture_is_flagged(path: Path) -> None:
    rid = rule_id_of(path)
    fired = run_fixture(path)
    assert rid in fired, f"{path.name}: expected {rid} to fire, got {sorted(fired)}"


@pytest.mark.parametrize("path", CLEAN_FILES, ids=lambda p: p.name)
def test_clean_fixture_not_flagged_for_its_rule(path: Path) -> None:
    rid = rule_id_of(path)
    fired = run_fixture(path)
    assert rid not in fired, f"{path.name}: {rid} fired on a clean example: {sorted(fired)}"


@pytest.mark.parametrize("path", CLEAN_FILES, ids=lambda p: p.name)
def test_clean_fixture_is_false_positive_free(path: Path) -> None:
    """The clean set should not trigger any rule."""
    fired = run_fixture(path)
    assert not fired, f"{path.name}: clean example produced findings {sorted(fired)}"


def test_every_catalog_rule_has_paired_fixtures() -> None:
    from leakproof.core.registry import all_rules

    leaky_ids = {rule_id_of(p) for p in LEAKY_FILES}
    clean_ids = {rule_id_of(p) for p in CLEAN_FILES}
    for rid in all_rules():
        assert rid in leaky_ids, f"missing leaky fixture for {rid}"
        assert rid in clean_ids, f"missing clean fixture for {rid}"
