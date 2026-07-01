"""Verify the plugin API: a registered rule runs, and a registered adapter
extends framework knowledge. We register directly (the entry-point mechanism is
exercised by the same registry path)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "example_plugin"))

import myorg_leakproof  # noqa: E402

from leakproof.core.config import Config  # noqa: E402
from leakproof.core.context import StaticContext  # noqa: E402
from leakproof.static.adapters import AdapterRegistry, load_adapters  # noqa: E402
from leakproof.static.dataflow import DataFlow  # noqa: E402


def _ctx(src, adapters):
    tree = ast.parse(src)
    lines = src.splitlines()
    return StaticContext(
        module_path=Path("m.py"),
        tree=tree,
        source=src,
        source_lines=lines,
        dataflow=DataFlow(tree, adapters, lines),
        adapters=adapters,
        config=Config(),
    )


def test_plugin_rule_fires():
    rule = myorg_leakproof.rules()[0]
    ctx = _ctx("x = eval('1+1')\n", load_adapters())
    findings = list(rule.check(ctx))
    assert findings and findings[0].rule_id == "myorg-X001"


def test_plugin_adapter_extends_transformers():
    base = load_adapters()
    extended = AdapterRegistry(base.adapters + [myorg_leakproof.adapter()])
    assert extended.is_transformer("MyCustomScaler")
    assert not base.is_transformer("MyCustomScaler")


def test_plugin_rule_id_is_vendor_prefixed():
    for rule in myorg_leakproof.rules():
        assert "-" in rule.id, "plugin rule ids must be vendor-prefixed"
