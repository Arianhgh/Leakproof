"""Determinism static rules: R001, R002."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.registry import register
from ...core.rule import StaticRule
from ._helpers import has_kwarg, kwarg_is_true, location_of, notebook_note

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
        global_seed_lines = self._global_seed_lines(ctx)
        seen: set[int] = set()
        for c in ctx.dataflow.calls:
            if not ctx.adapters.is_seeded(c.func_name):
                continue
            explicit_none = self._has_explicit_none_seed(c.node)
            if any(k in c.keywords for k in _SEED_KWARGS) and not explicit_none:
                continue
            if self._has_star_kwargs(c.node):
                continue
            if not self._call_needs_local_seed(c):
                continue
            # if a global np/random seed is set, downgrade train_test_split noise
            if (
                global_seed_lines
                and any(line <= c.line for line in global_seed_lines)
                and c.func_name not in ("train_test_split",)
                and not explicit_none
            ):
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
                advisory_only=True,
                fix=Fix(
                    summary=f"Add random_state=<int> to {c.func_name}.",
                    autofixable=True,
                ),
                evidence={"callable": c.func_name},
            )

    def _global_seed_lines(self, ctx: StaticContext) -> list[int]:
        lines: list[int] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                name = ctx.dataflow.resolve_name(node.func) or ""
                if name.endswith(("np.random.seed", "numpy.random.seed", "random.seed")):
                    lines.append(node.lineno)
                elif name.endswith(("manual_seed", "set_seed")):
                    lines.append(node.lineno)
        return lines

    def _has_explicit_none_seed(self, node: ast.Call) -> bool:
        return any(
            keyword.arg in _SEED_KWARGS
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is None
            for keyword in node.keywords
        )

    def _has_star_kwargs(self, node: ast.Call) -> bool:
        return any(kw.arg is None for kw in node.keywords)

    def _call_needs_local_seed(self, call: Any) -> bool:
        if call.func_name == "KMeans" and self._has_fixed_array_init(call.node):
            return False
        if call.func_name in {"KFold", "StratifiedKFold"}:
            return kwarg_is_true(has_kwarg(call.node, "shuffle"))
        if call.func_name == "LogisticRegression":
            solver = self._constant_kwarg(call.node, "solver")
            solver = solver or "lbfgs"
            return solver in {"liblinear", "sag", "saga"}
        if call.func_name == "SVC":
            return kwarg_is_true(has_kwarg(call.node, "probability"))
        return True

    def _has_fixed_array_init(self, node: ast.Call) -> bool:
        init = has_kwarg(node, "init")
        if init is None or isinstance(init, ast.Constant):
            return False
        n_init = has_kwarg(node, "n_init")
        return isinstance(n_init, ast.Constant) and n_init.value == 1

    def _constant_kwarg(self, node: ast.Call, name: str) -> Any:
        value = has_kwarg(node, name)
        if isinstance(value, ast.Constant):
            return value.value
        return None


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
        determinism_disabled = False
        first_torch_call = None
        determinism_settings: list[tuple[int, bool]] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                name = ctx.dataflow.resolve_name(node.func) or ""
                if (name == "torch" or name.startswith("torch.")) and first_torch_call is None:
                    first_torch_call = node
                if name.endswith("manual_seed"):
                    has_seed = True
                if name.endswith("use_deterministic_algorithms"):
                    value = _first_bool_argument(node)
                    if value is not None:
                        determinism_settings.append((node.lineno, value))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = ctx.dataflow.resolve_name(target) or ""
                    if name.lower().endswith("cudnn.deterministic"):
                        value = _literal_bool(node.value) if node.value is not None else None
                        if value is not None:
                            determinism_settings.append((node.lineno, value))
        if determinism_settings:
            has_determinism = sorted(determinism_settings)[-1][1]
            determinism_disabled = not has_determinism
        if first_torch_call is not None and not (has_seed and has_determinism):
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    "Torch is used without both torch.manual_seed(...) and an enabled "
                    "deterministic-algorithms setting; training may be nondeterministic."
                    + notebook_note(ctx)
                ),
                location=location_of(first_torch_call, ctx),
                confidence=0.5,
                references=(),
                advisory_only=True,
                fix=Fix(
                    summary="Set torch.manual_seed and torch.use_deterministic_algorithms(True).",
                    autofixable=False,
                ),
                evidence={
                    "has_seed": has_seed,
                    "has_determinism": has_determinism,
                    "determinism_disabled": determinism_disabled,
                },
            )


def _literal_bool(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _first_bool_argument(node: ast.Call) -> bool | None:
    if node.args:
        return _literal_bool(node.args[0])
    for keyword in node.keywords:
        if keyword.arg in {"mode", "enabled"}:
            return _literal_bool(keyword.value)
    return None
