# Writing a leakproof plugin

leakproof is extensible without forking: third parties contribute **rules** and
**framework adapters** via Python entry points. A plugin is a normal pip package.

## Public API

Everything you need is re-exported from `leakproof.plugin`:

```python
from leakproof.plugin import (
    register, register_adapter,
    Rule, StaticRule, DataRule, RuntimeRule,
    FrameworkAdapter,
    Finding, Fix, Severity, Category, Layer, Location,
)
```

## A rule

```python
import ast
from collections.abc import Iterable
from leakproof.plugin import StaticRule, Finding, Category, Severity, Layer, Location

class NoEval(StaticRule):
    id = "myorg-X001"          # MUST be vendor-prefixed to avoid collisions
    name = "calls forbidden eval()"
    category = Category.DETERMINISM
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    rationale = "Flag eval() in ML code."

    def check(self, ctx) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
                yield Finding(
                    rule_id=self.id, category=self.category, severity=self.severity,
                    layer=Layer.STATIC, message="eval() is forbidden.",
                    location=Location(file=ctx.module_path, line=node.lineno),
                )

def rules():
    return [NoEval()]
```

`StaticRule.check(ctx)` receives a `StaticContext` with `tree`, `source_lines`,
`dataflow` (the def-use/taint graph), and `adapters`. `DataRule.check(ctx)` receives a
`DataContext` with the `DataAuditInput`. `RuntimeRule.on_event(event, ctx)` receives each
runtime event.

## An adapter

```python
from leakproof.plugin import FrameworkAdapter

def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="myorg",
        transformers={"MyCustomScaler"},
        split_functions={"my_split"},
    )
```

## Register via entry points

```toml
# pyproject.toml of your plugin package
[project.entry-points."leakproof.rules"]
myorg = "myorg_leakproof:rules"

[project.entry-points."leakproof.adapters"]
myorg = "myorg_leakproof:adapter"
```

Install the plugin alongside leakproof and its rules/adapters load automatically.

A complete working example lives in `tests/integration/example_plugin/`.
