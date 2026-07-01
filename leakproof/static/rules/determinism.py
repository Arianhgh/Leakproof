"""Determinism static rules: R001, R002."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.registry import register
from ...core.rule import StaticRule
from ._helpers import location_of, notebook_note

_SEED_KWARGS = ("random_state", "seed", "random_seed")


@register
class R001(StaticRule):
    id = "R001"
    name = "missing random_state/seed"
    category = Category.DETERMINISM
    severity = Severity.LOW
    layers = (Layer.STATIC,)
    rationale = (
        "Without a fixed seed, splits and stochastic estimators vary run to run, making "
        "results irreproducible (and inviting seed cherry-picking)."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        global_seed = self._has_global_seed(ctx)
        seen: set[int] = set()
        for c in ctx.dataflow.calls:
            if not ctx.adapters.is_seeded(c.func_name):
                continue
            if any(k in c.keywords for k in _SEED_KWARGS):
                continue
            if self._has_star_kwargs(c.node):
                continue
            # if a global np/random seed is set, downgrade train_test_split noise
            if global_seed and c.func_name not in ("train_test_split",):
                continue
            if id(c.node) in seen:
                continue
            seen.add(id(c.node))
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    f"`{c.func_name}(...)` has no random_state/seed; fix it for reproducibility."
                    + notebook_note(ctx)
                ),
                location=location_of(c.node, ctx),
                confidence=0.7,
                references=(),
                fix=Fix(
                    summary=f"Add random_state=<int> to {c.func_name}.",
                    autofixable=True,
                ),
                evidence={"callable": c.func_name},
            )

    def _has_global_seed(self, ctx: StaticContext) -> bool:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                name = ctx.dataflow.resolve_name(node.func) or ""
                if name.endswith(("np.random.seed", "numpy.random.seed", "random.seed")):
                    return True
                if name.endswith(("manual_seed", "set_seed")):
                    return True
        return False

    def _has_star_kwargs(self, node: ast.Call) -> bool:
        return any(kw.arg is None for kw in node.keywords)


@register
class R002(StaticRule):
    id = "R002"
    name = "nondeterministic framework ops"
    category = Category.DETERMINISM
    severity = Severity.LOW
    layers = (Layer.STATIC,)
    rationale = (
        "Torch/cuDNN operations are nondeterministic by default. Set a seed and "
        "torch.use_deterministic_algorithms(True) for reproducible training."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        # require an actual top-level torch package import (not a relative
        # submodule that happens to be named "torch").
        uses_torch = any(
            v == "torch" or v.startswith("torch.") for v in ctx.dataflow.imports.values()
        )
        if not uses_torch:
            return
        has_seed = False
        has_determinism = False
        first_torch_call = None
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                name = ctx.dataflow.resolve_name(node.func) or ""
                if (name == "torch" or name.startswith("torch.")) and first_torch_call is None:
                    first_torch_call = node
                if name.endswith("manual_seed"):
                    has_seed = True
                if name.endswith("use_deterministic_algorithms") or "cudnn" in name.lower():
                    has_determinism = True
        if first_torch_call is not None and not (has_seed and has_determinism):
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    "Torch is used without both torch.manual_seed(...) and "
                    "torch.use_deterministic_algorithms(True); training is nondeterministic."
                    + notebook_note(ctx)
                ),
                location=location_of(first_torch_call, ctx),
                confidence=0.5,
                references=(),
                fix=Fix(
                    summary="Set torch.manual_seed and torch.use_deterministic_algorithms(True).",
                    autofixable=False,
                ),
                evidence={"has_seed": has_seed, "has_determinism": has_determinism},
            )
