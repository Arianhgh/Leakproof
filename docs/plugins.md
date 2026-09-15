# Writing an `ml-leakproof` plugin

Leakproof is extensible through Python entry points. Plugins are ordinary pip
packages; they do not need to fork or modify the built-in rule catalog.

## Public API

Import extension types from `ml_leakproof.plugin`:

```python
from ml_leakproof.plugin import (
    Category,
    DataRule,
    Finding,
    Fix,
    FrameworkAdapter,
    Layer,
    Location,
    RuntimeRule,
    Severity,
    StaticRule,
)
```

## A static rule

```python
import ast
from collections.abc import Iterable

from ml_leakproof.plugin import Category, Finding, Layer, Location, Severity, StaticRule


class NoEval(StaticRule):
    id = "myorg-X001"  # Vendor prefix is required for plugin IDs.
    name = "calls forbidden eval()"
    category = Category.DETERMINISM
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    rationale = "Flag eval() in analyzed source."

    def check(self, ctx) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "eval":
                    yield Finding(
                        rule_id=self.id,
                        category=self.category,
                        severity=self.severity,
                        layer=Layer.STATIC,
                        message="eval() is forbidden.",
                        location=Location(file=ctx.module_path, line=node.lineno),
                    )


def rules():
    return [NoEval]
```

The callable may return rule classes, instances, or an iterable of either. Each
entry point is validated and committed atomically: if one rule in a plugin batch
is invalid, none of that batch is loaded. Rule IDs must contain a vendor prefix
such as `myorg-`; invalid plugin loads are reported as operational diagnostics,
not hidden.

`StaticRule.check(ctx)` receives the AST, source lines, scope-aware dataflow,
and adapters. `DataRule.check(ctx)` receives a validated `DataContext`.
`RuntimeRule.on_event(event, ctx)` receives runtime events from an isolated
session. Keep plugin checks bounded and report unavailable work through the
result diagnostics where appropriate.

## A framework adapter

```python
from ml_leakproof.plugin import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="myorg",
        transformers={"myorg.preprocessing.MyScaler"},
        split_functions={"myorg.data.split"},
    )
```

Qualified names are matched conservatively. Bare symbols supplied by plugins
must be exact; broad suffix matching is reserved for known framework roots.

## Entry points

```toml
[project.entry-points."ml_leakproof.rules"]
myorg = "myorg_leakproof:rules"

[project.entry-points."ml_leakproof.adapters"]
myorg = "myorg_leakproof:adapter"
```

After installing the plugin, run `ml-leakproof rules --format json` to verify
the rule and its declared layers. The package and entry-point group names are
the 0.2 names; pre-0.2 `leakproof.*` imports and `leakproof.rules` groups are
not compatibility aliases.
