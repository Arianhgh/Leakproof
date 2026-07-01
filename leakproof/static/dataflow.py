"""Def-use and taint analysis for static leakage rules.

The static rules need more context than "a fit call appears near a split call." This
module records, per scope, where symbols came from:

  FULL    - data that predates a split, or combines both sides of one
  TRAIN   - derived from the training side only
  TEST    - derived from the test side only
  VAL     - derived from the validation side only
  UNKNOWN - not resolved by the lightweight analysis

Rules consume the structured records produced here: fit calls, split events, concats,
pipeline membership, and generic call records. The analysis is intentionally
intra-procedural; imports are resolved module-wide, but each function body is analyzed in
its own scope.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters import AdapterRegistry


class Taint(str, Enum):
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    UNKNOWN = "unknown"

    @property
    def is_eval_side(self) -> bool:
        return self in (Taint.TEST, Taint.VAL)


@dataclass
class FitCall:
    """A call to a learning method (fit/fit_transform/...)."""

    node: ast.Call
    line: int
    col: int
    method: str
    class_name: str | None  # resolved estimator/transformer class if known
    receiver_var: str | None  # variable the object is bound to, if any
    arg_vars: list[str]  # variable names passed as positional data args
    arg_taints: list[Taint]
    output_vars: list[str]  # variables the call result is assigned to, if any
    output_taints: list[Taint]
    in_pipeline: bool  # transformer is inside a Pipeline/make_pipeline
    in_cv: bool  # the fit happens via a CV utility (cross_val_score, ...)
    is_transformer: bool
    is_estimator: bool
    is_resampler: bool
    is_feature_selector: bool


@dataclass
class SplitEvent:
    node: ast.Call
    line: int
    func_name: str
    output_vars: list[str]
    output_taints: list[Taint]
    kwargs: dict[str, ast.expr]
    input_vars: list[str]


@dataclass
class ConcatAssign:
    target: str
    line: int
    source_vars: list[str]
    node: ast.Call


@dataclass
class CallRecord:
    """A generic resolved call (for rules that scan constructors/metrics)."""

    node: ast.Call
    line: int
    func_name: str  # resolved tail name
    keywords: dict[str, ast.expr]
    arg_vars: list[str]
    arg_taints: list[Taint]


@dataclass
class ScopeFlow:
    name: str
    taints: dict[str, Taint] = field(default_factory=dict)
    fit_calls: list[FitCall] = field(default_factory=list)
    splits: list[SplitEvent] = field(default_factory=list)
    concats: list[ConcatAssign] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    # var -> set of class names of objects bound (StandardScaler() -> {StandardScaler})
    var_classes: dict[str, str] = field(default_factory=dict)
    # vars that are transformer objects placed inside a pipeline
    pipelined_vars: set[str] = field(default_factory=set)


class DataFlow:
    """Whole-module imports plus one flow graph per executable scope."""

    def __init__(self, tree: ast.AST, adapters: AdapterRegistry, source_lines: list[str]):
        self.tree = tree
        self.adapters = adapters
        self.source_lines = source_lines
        self.imports: dict[str, str] = {}  # local name -> qualified
        self.scopes: list[ScopeFlow] = []
        self._call_targets: dict[int, list[str]] = {}  # id(Call rhs) -> target vars
        self._collect_imports()
        self._analyze()

    def _collect_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self.imports[local] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # mark relative imports with leading dots so a local submodule
                # named e.g. ``torch`` is not mistaken for the torch package.
                prefix = "." * (node.level or 0)
                for alias in node.names:
                    local = alias.asname or alias.name
                    qualified = f"{module}.{alias.name}" if module else alias.name
                    self.imports[local] = prefix + qualified

    def resolve_name(self, node: ast.expr) -> str | None:
        """Return a dotted name for a Name/Attribute callee."""
        if isinstance(node, ast.Name):
            return self.imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr | None = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                base = self.imports.get(cur.id, cur.id)
                parts.append(base)
                return ".".join(reversed(parts))
            parts.reverse()
            return ".".join(parts)
        return None

    @staticmethod
    def _tail(name: str | None) -> str | None:
        return name.rsplit(".", 1)[-1] if name else None

    def _analyze(self) -> None:
        module_scope = ScopeFlow(name="<module>")
        self.scopes.append(module_scope)
        self._analyze_body(self.tree, module_scope)

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = ScopeFlow(name=node.name)
                self.scopes.append(scope)
                fake = ast.Module(body=node.body, type_ignores=[])
                self._analyze_body(fake, scope)

    def _analyze_body(self, tree: ast.AST, scope: ScopeFlow) -> None:
        nodes = list(_scope_nodes(tree))

        self._collect_pipeline_membership(nodes, scope)
        for stmt in nodes:
            if isinstance(stmt, ast.Assign):
                self._handle_assign(stmt, scope)
        for node in nodes:
            if isinstance(node, ast.Call):
                self._record_call(node, scope)

    def _collect_pipeline_membership(self, nodes: list[ast.AST], scope: ScopeFlow) -> None:
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            name = self.resolve_name(node.func)
            if name is None or not self.adapters.is_pipeline_constructor(name):
                continue
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Name):
                        scope.pipelined_vars.add(sub.id)
            for kw in node.keywords:
                for sub in ast.walk(kw.value):
                    if isinstance(sub, ast.Name):
                        scope.pipelined_vars.add(sub.id)

    def _handle_assign(self, stmt: ast.Assign, scope: ScopeFlow) -> None:
        rhs = stmt.value
        if isinstance(rhs, ast.Call):
            self._call_targets[id(rhs)] = _assign_targets(stmt)
        if isinstance(rhs, ast.Call):
            fname = self.resolve_name(rhs.func)
            if fname and self.adapters.is_split_function(fname):
                self._handle_split_assign(stmt, rhs, fname, scope)
                return
            if self._is_concat(rhs):
                self._handle_concat(stmt, rhs, scope)
            ctor = self.resolve_name(rhs.func)
            if ctor:
                tail = self._tail(ctor)
                if tail and (
                    self.adapters.is_transformer(ctor)
                    or self.adapters.is_estimator(ctor)
                    or self.adapters.is_resampler(ctor)
                    or self.adapters.is_feature_selector(ctor)
                    or self.adapters.is_search_constructor(ctor)
                    or self.adapters.is_pipeline_constructor(ctor)
                ):
                    for tgt in _assign_targets(stmt):
                        scope.var_classes[tgt] = tail

        taint = self._infer_taint(rhs, scope)
        for tgt in _assign_targets(stmt):
            if tgt in scope.taints and scope.taints[tgt] in (Taint.TRAIN, Taint.TEST, Taint.VAL):
                continue
            scope.taints[tgt] = taint

    def _handle_split_assign(
        self, stmt: ast.Assign, call: ast.Call, fname: str, scope: ScopeFlow
    ) -> None:
        targets: list[str] = []
        for t in stmt.targets:
            if isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    targets.append(elt.id if isinstance(elt, ast.Name) else "?")
            elif isinstance(t, ast.Name):
                targets.append(t.id)

        out_taints = _split_output_taints(targets)
        for name, taint in zip(targets, out_taints):
            if name != "?":
                scope.taints[name] = taint

        input_vars = [a.id for a in call.args if isinstance(a, ast.Name)]
        for v in input_vars:
            if scope.taints.get(v) not in (Taint.TRAIN, Taint.TEST, Taint.VAL):
                scope.taints[v] = Taint.FULL
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        scope.splits.append(
            SplitEvent(
                node=call,
                line=call.lineno,
                func_name=self._tail(fname) or fname,
                output_vars=targets,
                output_taints=out_taints,
                kwargs=kwargs,
                input_vars=input_vars,
            )
        )

    def _is_concat(self, call: ast.Call) -> bool:
        name = self.resolve_name(call.func)
        tail = self._tail(name)
        return tail in {"concat", "vstack", "hstack", "append", "merge", "concatenate"}

    def _handle_concat(self, stmt: ast.Assign, call: ast.Call, scope: ScopeFlow) -> None:
        source_vars: list[str] = []
        for arg in call.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                for elt in arg.elts:
                    if isinstance(elt, ast.Name):
                        source_vars.append(elt.id)
            elif isinstance(arg, ast.Name):
                source_vars.append(arg.id)
        for tgt in _assign_targets(stmt):
            scope.concats.append(
                ConcatAssign(target=tgt, line=call.lineno, source_vars=source_vars, node=call)
            )
            scope.taints[tgt] = self._concat_taint(source_vars, scope)

    def _concat_taint(self, source_vars: list[str], scope: ScopeFlow) -> Taint:
        seen = {scope.taints.get(v, Taint.UNKNOWN) for v in source_vars}
        sides = {t for t in seen if t in (Taint.TRAIN, Taint.TEST, Taint.VAL)}
        if len(sides) >= 2:
            return Taint.FULL
        if Taint.FULL in seen:
            return Taint.FULL
        if len(sides) == 1:
            return next(iter(sides))
        return Taint.UNKNOWN

    def _infer_taint(self, expr: ast.expr, scope: ScopeFlow) -> Taint:
        """Propagate taint through simple derivations."""
        names = [n.id for n in ast.walk(expr) if isinstance(n, ast.Name)]
        labels = {scope.taints.get(n) for n in names if n in scope.taints}
        labels.discard(None)
        if not labels:
            if isinstance(expr, ast.Name):
                return self._name_taint(expr.id)
            return Taint.UNKNOWN
        sides = {t for t in labels if t in (Taint.TRAIN, Taint.TEST, Taint.VAL)}
        if Taint.FULL in labels:
            return Taint.FULL
        if len(sides) >= 2:
            return Taint.FULL
        if len(sides) == 1:
            return next(iter(sides))
        return Taint.UNKNOWN

    def _record_call(self, node: ast.Call, scope: ScopeFlow) -> None:
        if not isinstance(node.func, ast.Attribute):
            fname = self.resolve_name(node.func)
            if fname:
                scope.calls.append(self._make_call_record(node, fname, scope))
            return

        method = node.func.attr
        receiver = node.func.value
        receiver_var = receiver.id if isinstance(receiver, ast.Name) else None

        class_name = self._resolve_receiver_class(receiver, scope)
        scope.calls.append(self._make_call_record(node, method, scope))

        if not self.adapters.is_learn_method(method):
            return

        arg_vars = [a.id for a in node.args if isinstance(a, ast.Name)]
        arg_taints = [scope.taints.get(v, self._name_taint(v)) for v in arg_vars]
        output_vars = self._call_targets.get(id(node), [])
        output_taints = [scope.taints.get(v, self._name_taint(v)) for v in output_vars]

        in_pipeline = bool(receiver_var and receiver_var in scope.pipelined_vars)
        if isinstance(receiver, ast.Call):
            class_name = self._tail(self.resolve_name(receiver.func)) or class_name

        scope.fit_calls.append(
            FitCall(
                node=node,
                line=node.lineno,
                col=node.col_offset,
                method=method,
                class_name=class_name,
                receiver_var=receiver_var,
                arg_vars=arg_vars,
                arg_taints=arg_taints,
                output_vars=output_vars,
                output_taints=output_taints,
                in_pipeline=in_pipeline,
                in_cv=False,
                is_transformer=bool(class_name and self.adapters.is_transformer(class_name)),
                is_estimator=bool(class_name and self.adapters.is_estimator(class_name)),
                is_resampler=bool(class_name and self.adapters.is_resampler(class_name)),
                is_feature_selector=bool(
                    class_name and self.adapters.is_feature_selector(class_name)
                ),
            )
        )

    def _make_call_record(self, node: ast.Call, fname: str, scope: ScopeFlow) -> CallRecord:
        arg_vars = [a.id for a in node.args if isinstance(a, ast.Name)]
        arg_taints = [scope.taints.get(v, self._name_taint(v)) for v in arg_vars]
        return CallRecord(
            node=node,
            line=node.lineno,
            func_name=self._tail(fname) or fname,
            keywords={kw.arg: kw.value for kw in node.keywords if kw.arg},
            arg_vars=arg_vars,
            arg_taints=arg_taints,
        )

    def _resolve_receiver_class(self, receiver: ast.expr, scope: ScopeFlow) -> str | None:
        if isinstance(receiver, ast.Name):
            if receiver.id in scope.var_classes:
                return scope.var_classes[receiver.id]
            return None
        if isinstance(receiver, ast.Call):
            return self._tail(self.resolve_name(receiver.func))
        return None

    @staticmethod
    def _name_taint(var: str) -> Taint:
        """Last-resort taint from a variable's name suffix."""
        low = var.lower()
        if low.endswith(("_test", "_te")) or low in ("x_test", "y_test", "test"):
            return Taint.TEST
        if low.endswith(("_val", "_valid")) or low in ("x_val", "y_val", "val"):
            return Taint.VAL
        if low.endswith("_train") or low in ("x_train", "y_train", "train"):
            return Taint.TRAIN
        return Taint.UNKNOWN

    @property
    def fit_calls(self) -> list[FitCall]:
        out: list[FitCall] = []
        for s in self.scopes:
            out.extend(s.fit_calls)
        return out

    @property
    def splits(self) -> list[SplitEvent]:
        out: list[SplitEvent] = []
        for s in self.scopes:
            out.extend(s.splits)
        return out

    @property
    def concats(self) -> list[ConcatAssign]:
        out: list[ConcatAssign] = []
        for s in self.scopes:
            out.extend(s.concats)
        return out

    @property
    def calls(self) -> list[CallRecord]:
        out: list[CallRecord] = []
        for s in self.scopes:
            out.extend(s.calls)
        return out

    @property
    def concat_targets(self) -> set[str]:
        out: set[str] = set()
        for s in self.scopes:
            out |= {c.target for c in s.concats}
        return out

    def has_split(self) -> bool:
        return bool(self.splits)

    @staticmethod
    def leak_taint(fit: FitCall) -> Taint | None:
        """Return the offending taint if this fit consumes/produces leaky data.

        A fit leaks when its input *or* its output derives from pre-split (FULL)
        data or from the eval side (TEST/VAL).
        """
        for t in [*fit.arg_taints, *fit.output_taints]:
            if t is Taint.FULL or t.is_eval_side:
                return t
        return None

    def fit_arg_is_concat(self, fit: FitCall) -> bool:
        targets = self.concat_targets
        return any(v in targets for v in fit.arg_vars)

    def taint_of(self, name: str) -> Taint:
        for s in self.scopes:
            if name in s.taints:
                return s.taints[name]
        return self._name_taint(name)

    def class_of(self, name: str) -> str | None:
        for s in self.scopes:
            if name in s.var_classes:
                return s.var_classes[name]
        return None

    def uses_cv_utility(self) -> bool:
        return any(self.adapters.is_search_constructor(c.func_name) for c in self.calls)

    def scope_of_line(self, line: int) -> ScopeFlow | None:
        for s in self.scopes:
            for rec in s.fit_calls:
                if rec.line == line:
                    return s
        return self.scopes[0] if self.scopes else None


_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _scope_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes for the current scope without descending into nested scopes."""
    initial = list(reversed(getattr(tree, "body", [tree])))
    stack: list[ast.AST] = initial
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_BOUNDARIES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _assign_targets(stmt: ast.Assign) -> list[str]:
    out: list[str] = []
    for t in stmt.targets:
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                if isinstance(elt, ast.Name):
                    out.append(elt.id)
    return out


def _split_output_taints(targets: list[str]) -> list[Taint]:
    """train_test_split returns [in0_train, in0_test, in1_train, in1_test, ...].

    Prefer explicit name suffixes; fall back to position parity (even=train,
    odd=test).
    """
    out: list[Taint] = []
    for i, name in enumerate(targets):
        by_name = DataFlow._name_taint(name)
        if by_name is not Taint.UNKNOWN:
            out.append(by_name)
        else:
            out.append(Taint.TRAIN if i % 2 == 0 else Taint.TEST)
    return out
