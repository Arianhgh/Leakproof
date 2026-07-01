"""Example leakproof plugin used by the integration tests.

A real plugin ships as a pip package declaring entry points::

    [project.entry-points."leakproof.rules"]
    myorg = "myorg_leakproof:rules"

    [project.entry-points."leakproof.adapters"]
    myorg = "myorg_leakproof:adapter"

Plugin rule IDs MUST be vendor-prefixed (here ``myorg-``) to avoid collisions.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from leakproof.plugin import (
    Category,
    Finding,
    FrameworkAdapter,
    Layer,
    Location,
    Severity,
    StaticRule,
)


class MyOrgNoEvalCall(StaticRule):
    id = "myorg-X001"
    name = "calls forbidden eval()"
    category = Category.DETERMINISM
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    rationale = "Example org rule: flag eval() calls in ML code."

    def check(self, ctx) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "eval"
            ):
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.STATIC,
                    message="eval() is forbidden by org policy.",
                    location=Location(file=ctx.module_path, line=node.lineno),
                )


def rules():
    """Entry-point callable returning rule instances."""
    return [MyOrgNoEvalCall()]


def adapter() -> FrameworkAdapter:
    """Entry-point callable returning a framework adapter."""
    return FrameworkAdapter(name="myorg", transformers={"MyCustomScaler"})
