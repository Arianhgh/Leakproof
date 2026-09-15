"""Order-aware, scope-aware intermediate representation for Python source.

This is intentionally a bounded static analysis.  It tracks facts that are
useful for leakage evidence (bindings, row-side taint, estimator state, split
roles, and evaluation calls) without executing user code or pretending to solve
arbitrary Python data flow.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

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
    node: ast.Call
    line: int
    col: int
    method: str
    class_name: str | None
    receiver_var: str | None
    arg_vars: list[str]
    arg_taints: list[Taint]
    output_vars: list[str]
    output_taints: list[Taint]
    in_pipeline: bool
    in_cv: bool
    is_transformer: bool
    is_estimator: bool
    is_resampler: bool
    is_feature_selector: bool
    keyword_vars: dict[str, list[str]] = field(default_factory=dict)
    scope_name: str = "<module>"
    scope_id: int = 0
    arg_binding_versions: dict[str, int] = field(default_factory=dict)
    output_binding_versions: dict[str, int] = field(default_factory=dict)
    receiver_binding_version: int | None = None


@dataclass
class SplitEvent:
    node: ast.Call
    line: int
    func_name: str
    output_vars: list[str]
    output_taints: list[Taint]
    kwargs: dict[str, ast.expr]
    input_vars: list[str]
    split_id: str = "split-0"
    roles: dict[str, str] = field(default_factory=dict)
    scope_id: int = 0
    input_binding_versions: dict[str, int] = field(default_factory=dict)


@dataclass
class ConcatAssign:
    target: str
    line: int
    source_vars: list[str]
    node: ast.Call
    axis: int | None = None
    operation: str = "concat"
    scope_id: int = 0
    source_binding_versions: dict[str, int] = field(default_factory=dict)
    target_binding_version: int = 0


@dataclass
class CallRecord:
    node: ast.Call
    line: int
    func_name: str
    keywords: dict[str, ast.expr]
    arg_vars: list[str]
    arg_taints: list[Taint]
    keyword_vars: dict[str, list[str]] = field(default_factory=dict)
    scope_name: str = "<module>"
    scope_id: int = 0
    receiver_var: str | None = None
    receiver_binding_version: int | None = None
    arg_binding_versions: dict[str, int] = field(default_factory=dict)
    output_binding_versions: dict[str, int] = field(default_factory=dict)


@dataclass
class ScopeFlow:
    name: str
    scope_id: int = 0
    taints: dict[str, Taint] = field(default_factory=dict)
    fit_calls: list[FitCall] = field(default_factory=list)
    splits: list[SplitEvent] = field(default_factory=list)
    concats: list[ConcatAssign] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    var_classes: dict[str, str] = field(default_factory=dict)
    pipelined_vars: set[str] = field(default_factory=set)
    imports: dict[str, str] = field(default_factory=dict)
    pipeline_bindings: set[str] = field(default_factory=set)
    assignment_lines: dict[str, int] = field(default_factory=dict)
    binding_versions: dict[str, int] = field(default_factory=dict)
    next_binding_version: int = 0


class DataFlow:
    """Build a bounded IR while preserving source execution order."""

    def __init__(self, tree: ast.AST, adapters: AdapterRegistry, source_lines: list[str]):
        self.tree = tree
        self.adapters = adapters
        self.source_lines = source_lines
        self.imports: dict[str, str] = {}
        self.scopes: list[ScopeFlow] = []
        self._call_targets: dict[int, list[str]] = {}
        self._node_scopes: dict[int, ScopeFlow] = {}
        self._split_index = 0
        self._analyze()

    def _analyze(self) -> None:
        module = ScopeFlow(name="<module>", scope_id=0)
        self.scopes.append(module)
        self._analyze_body(getattr(self.tree, "body", []), module)
        self.imports = dict(module.imports)

        # Function bodies are analyzed independently.  We copy module imports
        # as global bindings, then allow local imports/assignments to shadow them.
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = ScopeFlow(
                    name=node.name,
                    scope_id=len(self.scopes),
                    imports=dict(module.imports),
                )
                self.scopes.append(scope)
                for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    self._assign_name(argument.arg, Taint.UNKNOWN, scope, node.lineno)
                if node.args.vararg:
                    self._assign_name(node.args.vararg.arg, Taint.UNKNOWN, scope, node.lineno)
                if node.args.kwarg:
                    self._assign_name(node.args.kwarg.arg, Taint.UNKNOWN, scope, node.lineno)
                self._analyze_body(node.body, scope)
        self._retrofit_pre_split_full_inputs()

    def _analyze_body(self, statements: list[ast.stmt], scope: ScopeFlow) -> None:
        for stmt in statements:
            self._visit_statement(stmt, scope)

    def _visit_statement(self, stmt: ast.stmt, scope: ScopeFlow) -> None:
        self._node_scopes[id(stmt)] = scope
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                scope.imports[local] = alias.name
            return
        if isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ""
            prefix = "." * (stmt.level or 0)
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                scope.imports[local] = f"{prefix}{module}.{alias.name}" if module else alias.name
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A local definition shadows an imported framework name.  Its body
            # is visited in a separate bounded scope above.
            name = getattr(stmt, "name", None)
            if name:
                scope.imports.pop(name, None)
                self._assign_name(name, Taint.UNKNOWN, scope, stmt.lineno)
            return
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = stmt.value
            if not isinstance(value, ast.expr):
                return
            targets = _assignment_targets(stmt)
            if isinstance(value, ast.Call):
                self._call_targets[id(value)] = targets
            self._visit_expr(value, scope)
            self._finish_assignment(targets, value, scope, getattr(stmt, "lineno", 0))
            return
        if isinstance(stmt, ast.AugAssign):
            self._visit_expr(stmt.value, scope)
            target_names = _target_names(stmt.target)
            merged = [scope.taints.get(name, self._name_taint(name)) for name in target_names]
            merged.append(self._infer_taint(stmt.value, scope))
            taint = self._merge_taints(merged)
            for name in target_names:
                self._assign_name(name, taint, scope, stmt.lineno)
            return

        # Visit control-flow statements conservatively.  We do not invent facts
        # from a branch: assignments are merged as UNKNOWN when branches disagree.
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            self._visit_expr(getattr(stmt, "iter", getattr(stmt, "test", ast.Constant(None))), scope)
            before = dict(scope.taints)
            for child in stmt.body:
                self._visit_statement(child, scope)
            body_values = dict(scope.taints)
            scope.taints = _merge_branch_taints(before, body_values)
            for child in stmt.orelse:
                self._visit_statement(child, scope)
            return
        if isinstance(stmt, ast.If):
            self._visit_expr(stmt.test, scope)
            before = dict(scope.taints)
            for child in stmt.body:
                self._visit_statement(child, scope)
            body_values = dict(scope.taints)
            scope.taints = dict(before)
            for child in stmt.orelse:
                self._visit_statement(child, scope)
            else_values = dict(scope.taints)
            scope.taints = _merge_branch_taints(body_values, else_values)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._visit_expr(item.context_expr, scope)
                if item.optional_vars:
                    for name in _target_names(item.optional_vars):
                        self._assign_name(name, Taint.UNKNOWN, scope, stmt.lineno)
            for child in stmt.body:
                self._visit_statement(child, scope)
            return
        if isinstance(stmt, ast.Try):
            before = dict(scope.taints)
            for child in stmt.body:
                self._visit_statement(child, scope)
            branches = [dict(scope.taints)]
            for handler in stmt.handlers:
                scope.taints = dict(before)
                for child in handler.body:
                    self._visit_statement(child, scope)
                branches.append(dict(scope.taints))
            scope.taints = _merge_many(branches)
            for final_stmt in stmt.finalbody:
                self._visit_statement(final_stmt, scope)
            return

        for child_node in ast.iter_child_nodes(stmt):
            if isinstance(child_node, ast.expr):
                self._visit_expr(child_node, scope)
            elif isinstance(child_node, ast.stmt):
                self._visit_statement(child_node, scope)

    def _visit_expr(self, expr: ast.expr, scope: ScopeFlow) -> None:
        self._node_scopes[id(expr)] = scope
        if isinstance(expr, ast.Call):
            # Python evaluates the callee and arguments before invoking the call.
            self._visit_expr(expr.func, scope)
            for arg in expr.args:
                self._visit_expr(arg, scope)
            for kw in expr.keywords:
                self._visit_expr(kw.value, scope)
            self._record_call(expr, scope)
            return
        if isinstance(expr, ast.Lambda):
            return
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                self._visit_expr(child, scope)

    def _finish_assignment(
        self, targets: list[str], value: ast.expr, scope: ScopeFlow, line: int
    ) -> None:
        constructor_class: str | None = None
        if isinstance(value, ast.Call):
            name = self.resolve_name(value.func)
            if name and self.adapters.is_split_function(name):
                self._handle_split_assign(targets, value, name, scope)
                return
            if self._is_concat(value, scope):
                self._handle_concat(targets, value, scope)
            constructor_class = self._tail(name) if name and self._is_known_constructor(name) else None
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "fit_transform",
                "fit_resample",
            }:
                # The result remains on the same row side as its source.  The
                # fitted receiver is tracked separately in FitCall.
                pass

        taint = self._infer_taint(value, scope)
        for target in targets:
            self._assign_name(target, taint, scope, line)
        for fit in scope.fit_calls:
            if fit.node is value:
                fit.output_binding_versions = self._binding_versions(targets, scope)
        for call in scope.calls:
            if call.node is value:
                call.output_binding_versions = self._binding_versions(targets, scope)
        for concat in scope.concats:
            if concat.node is value:
                concat.target_binding_version = scope.binding_versions.get(concat.target, 0)
        # Aliasing an estimator (or an attribute-held estimator) retains class
        # identity, but reassignment always clears stale class facts first.
        if isinstance(value, ast.Name) and value.id in scope.var_classes:
            for target in targets:
                scope.var_classes[target] = scope.var_classes[value.id]
        elif constructor_class:
            for target in targets:
                scope.var_classes[target] = constructor_class

    def _assign_name(self, name: str, taint: Taint, scope: ScopeFlow, line: int) -> None:
        if not name or name == "?":
            return
        scope.taints[name] = taint
        scope.assignment_lines[name] = line
        scope.next_binding_version += 1
        scope.binding_versions[name] = scope.next_binding_version
        scope.var_classes.pop(name, None)
        # An assignment to an imported local shadows the import.  Qualified
        # module aliases are allowed to be rebound too.
        if name in scope.imports:
            scope.imports.pop(name, None)

    def _handle_split_assign(
        self, targets: list[str], call: ast.Call, fname: str, scope: ScopeFlow
    ) -> None:
        input_exprs = list(call.args)
        # ``train_test_split`` and compatible splitters are commonly called
        # with ``X=``/``y=``.  Treat those values as the split source too;
        # configuration keywords such as ``random_state`` must not become
        # fabricated data lineage.
        input_exprs.extend(
            keyword.value
            for keyword in call.keywords
            if keyword.arg in {"X", "x", "data", "y", "target", "labels", "stratify", "groups"}
        )
        input_taint = self._merge_taints(
            [self._infer_taint(expr, scope) for expr in input_exprs]
        )
        out_taints = _split_output_taints(len(targets), input_taint)
        input_vars: list[str] = []
        for expr in input_exprs:
            input_vars.extend(_expr_names(expr))
        input_vars = list(dict.fromkeys(input_vars))
        input_binding_versions = self._binding_versions(input_vars, scope)
        for name, taint in zip(targets, out_taints):
            self._assign_name(name, taint, scope, call.lineno)
        for var in input_vars:
            if scope.taints.get(var, Taint.UNKNOWN) not in {
                Taint.TRAIN,
                Taint.TEST,
                Taint.VAL,
            }:
                scope.taints[var] = Taint.FULL
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        split_id = f"split-{self._split_index}"
        self._split_index += 1
        roles = {
            name: taint.value
            for name, taint in zip(targets, out_taints)
            if name and name != "?"
        }
        scope.splits.append(
            SplitEvent(
                node=call,
                line=call.lineno,
                func_name=self._tail(fname) or fname,
                output_vars=list(targets),
                output_taints=out_taints,
                kwargs=kwargs,
                input_vars=input_vars,
                split_id=split_id,
                roles=roles,
                scope_id=scope.scope_id,
                input_binding_versions=input_binding_versions,
            )
        )

    def _is_concat(self, call: ast.Call, scope: ScopeFlow) -> bool:
        name = self.resolve_name(call.func)
        tail = self._tail(name) or (call.func.attr if isinstance(call.func, ast.Attribute) else None)
        # append/merge on an arbitrary object are not automatically row
        # concatenation.  pandas/numpy qualified functions are supported.
        if tail not in {"concat", "vstack", "hstack", "concatenate"}:
            return False
        if name is None:
            return False
        return name.split(".", 1)[0] in {"pandas", "numpy", "np", "pd"}

    def _handle_concat(self, targets: list[str], call: ast.Call, scope: ScopeFlow) -> None:
        source_vars: list[str] = []
        for arg in call.args:
            source_vars.extend(_expr_names(arg))
        axis = _constant_int(next((kw.value for kw in call.keywords if kw.arg == "axis"), None))
        operation = self._tail(self.resolve_name(call.func)) or "concat"
        taint = self._merge_taints([scope.taints.get(v, Taint.UNKNOWN) for v in source_vars])
        source_binding_versions = self._binding_versions(source_vars, scope)
        for target in targets:
            scope.concats.append(
                ConcatAssign(
                    target=target,
                    line=call.lineno,
                    source_vars=source_vars,
                    node=call,
                    axis=axis,
                    operation=operation,
                    scope_id=scope.scope_id,
                    source_binding_versions=source_binding_versions,
                )
            )
            self._assign_name(target, taint, scope, call.lineno)

    def _record_call(self, node: ast.Call, scope: ScopeFlow) -> None:
        self._node_scopes[id(node)] = scope
        func_name: str | None = None
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver = node.func.value
            receiver_var = _expr_ref(receiver)
            class_name = self._resolve_receiver_class(receiver, scope)
            func_name = method
        else:
            method = None
            receiver_var = None
            class_name = None
            func_name = self.resolve_name(node.func)
            if func_name is None:
                return

        resolved_func = self.resolve_name(node.func) or func_name
        call_name = self._tail(resolved_func) or resolved_func
        arg_vars = _expr_names_from_args(node.args)
        keyword_vars = {
            kw.arg: _expr_names(kw.value) for kw in node.keywords if kw.arg is not None
        }
        all_vars = list(arg_vars)
        for names in keyword_vars.values():
            all_vars.extend(names)
        arg_taints = [self._infer_taint(expr, scope) for expr in node.args]
        for kw in node.keywords:
            if kw.arg is not None:
                arg_taints.append(self._infer_taint(kw.value, scope))
        scope.calls.append(
            CallRecord(
                node=node,
                line=node.lineno,
                func_name=call_name,
                keywords={kw.arg: kw.value for kw in node.keywords if kw.arg},
                arg_vars=all_vars,
                arg_taints=arg_taints,
                keyword_vars=keyword_vars,
                scope_name=scope.name,
                scope_id=scope.scope_id,
                receiver_var=receiver_var,
                receiver_binding_version=(
                    scope.binding_versions.get(receiver_var, 0)
                    if receiver_var is not None
                    else None
                ),
                arg_binding_versions=self._binding_versions(all_vars, scope),
            )
        )

        if isinstance(node.func, ast.Attribute) and self.adapters.is_learn_method(method or ""):
            output_vars = self._call_targets.get(id(node), [])
            output_taints = [self._infer_taint(node, scope) for _ in output_vars]
            # A component being listed in a Pipeline is not evidence that a
            # separate ``component.fit(...)`` call is safe.  Only the
            # Pipeline/ColumnTransformer object's own fit call (or an inline
            # pipeline expression) is contextualized as pipeline-owned.
            in_pipeline = bool(receiver_var and receiver_var in scope.pipeline_bindings)
            if class_name is None and isinstance(receiver, ast.Call):
                ctor = self.resolve_name(receiver.func)
                if ctor:
                    class_name = self._tail(ctor)
                    in_pipeline = in_pipeline or self._is_pipeline_call(receiver, scope)
            fit = FitCall(
                node=node,
                line=node.lineno,
                col=node.col_offset,
                method=method or "",
                class_name=class_name,
                receiver_var=receiver_var,
                arg_vars=all_vars,
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
                keyword_vars=keyword_vars,
                scope_name=scope.name,
                scope_id=scope.scope_id,
                arg_binding_versions=self._binding_versions(all_vars, scope),
                receiver_binding_version=(
                    scope.binding_versions.get(receiver_var, 0)
                    if receiver_var is not None
                    else None
                ),
            )
            scope.fit_calls.append(fit)

        # Pipeline membership becomes true only after the constructor expression
        # is evaluated.  This preserves an earlier manual fit as unsafe.
        if self.adapters.is_pipeline_constructor(resolved_func):
            for name in _expr_names_from_args(node.args):
                scope.pipelined_vars.add(name)
            for names in keyword_vars.values():
                scope.pipelined_vars.update(names)
            for target in self._call_targets.get(id(node), []):
                scope.pipeline_bindings.add(target)
                scope.var_classes[target] = self._tail(resolved_func) or "Pipeline"

    def _resolve_receiver_class(self, receiver: ast.expr, scope: ScopeFlow) -> str | None:
        ref = _expr_ref(receiver)
        if ref and ref in scope.var_classes:
            return scope.var_classes[ref]
        if isinstance(receiver, ast.Call):
            ctor = self.resolve_name(receiver.func)
            if ctor:
                return self._tail(ctor)
        return None

    @staticmethod
    def _binding_versions(names: Iterable[str], scope: ScopeFlow) -> dict[str, int]:
        return {
            name: scope.binding_versions.get(name, 0)
            for name in dict.fromkeys(name for name in names if name and name != "?")
        }

    def _is_pipeline_call(self, receiver: ast.Call, scope: ScopeFlow) -> bool:
        name = self.resolve_name(receiver.func)
        return bool(name and self.adapters.is_pipeline_constructor(name))

    def _is_known_constructor(self, name: str) -> bool:
        return any(
            (
                self.adapters.is_transformer(name),
                self.adapters.is_estimator(name),
                self.adapters.is_resampler(name),
                self.adapters.is_feature_selector(name),
                self.adapters.is_search_constructor(name),
                self.adapters.is_pipeline_constructor(name),
            )
        )

    def resolve_name(self, node: ast.expr) -> str | None:
        """Resolve only imported/qualified names; bare local class names are unknown."""

        scope = self._node_scopes.get(id(node)) or (self.scopes[0] if self.scopes else None)
        imports = scope.imports if scope is not None else self.imports
        if isinstance(node, ast.Name):
            return imports.get(node.id)
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr | None = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                base = imports.get(cur.id)
                if base is None:
                    return None
                parts.append(base)
                return ".".join(reversed(parts))
        return None

    @staticmethod
    def _tail(name: str | None) -> str | None:
        return name.rsplit(".", 1)[-1] if name else None

    def _infer_taint(self, expr: ast.expr, scope: ScopeFlow) -> Taint:
        if isinstance(expr, ast.Name):
            return scope.taints.get(expr.id, self._name_taint(expr.id))
        if isinstance(expr, ast.Subscript):
            return self._infer_taint(expr.value, scope)
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Attribute):
                method = expr.func.attr
                if method in {"transform", "predict", "predict_proba", "decision_function"}:
                    input_taints = [self._infer_taint(arg, scope) for arg in expr.args]
                    input_taints.extend(self._infer_taint(kw.value, scope) for kw in expr.keywords)
                    if input_taints:
                        return self._merge_taints(input_taints)
                    return self._infer_taint(expr.func.value, scope)
                if method in {"fit_transform", "fit_resample", "resample"}:
                    return self._merge_taints(
                        [self._infer_taint(arg, scope) for arg in expr.args]
                        + [self._infer_taint(kw.value, scope) for kw in expr.keywords]
                    )
            return self._merge_taints(
                [self._infer_taint(arg, scope) for arg in expr.args]
                + [self._infer_taint(kw.value, scope) for kw in expr.keywords]
            )
        names = _expr_names(expr)
        return self._merge_taints([scope.taints.get(n, self._name_taint(n)) for n in names])

    @staticmethod
    def _merge_taints(taints: Iterable[Taint]) -> Taint:
        values = set(taints)
        sides = {value for value in values if value in {Taint.TRAIN, Taint.TEST, Taint.VAL}}
        if Taint.FULL in values or len(sides) >= 2:
            return Taint.FULL
        if len(sides) == 1:
            return next(iter(sides))
        return Taint.UNKNOWN

    @staticmethod
    def _name_taint(var: str) -> Taint:
        low = var.lower()
        if low.endswith(("_test", "_te")) or low in {"x_test", "y_test", "test"}:
            return Taint.TEST
        if low.endswith(("_val", "_valid")) or low in {"x_val", "y_val", "val"}:
            return Taint.VAL
        if low.endswith("_train") or low in {"x_train", "y_train", "train"}:
            return Taint.TRAIN
        return Taint.UNKNOWN

    @property
    def fit_calls(self) -> list[FitCall]:
        return sorted((fit for scope in self.scopes for fit in scope.fit_calls), key=_node_key)

    @property
    def splits(self) -> list[SplitEvent]:
        return sorted((split for scope in self.scopes for split in scope.splits), key=_node_key)

    @property
    def concats(self) -> list[ConcatAssign]:
        return sorted((concat for scope in self.scopes for concat in scope.concats), key=_node_key)

    @property
    def calls(self) -> list[CallRecord]:
        return sorted((call for scope in self.scopes for call in scope.calls), key=_node_key)

    @property
    def concat_targets(self) -> set[str]:
        return {concat.target for scope in self.scopes for concat in scope.concats}

    def has_split(self) -> bool:
        return bool(self.splits)

    @staticmethod
    def leak_taint(fit: FitCall) -> Taint | None:
        for taint in [*fit.arg_taints, *fit.output_taints]:
            if taint is Taint.FULL or taint.is_eval_side:
                return taint
        return None

    def fit_arg_is_concat(self, fit: FitCall) -> bool:
        for concat in self.concats:
            if concat.scope_id != fit.scope_id:
                continue
            for var in fit.arg_vars:
                if var == concat.target and fit.arg_binding_versions.get(var) == concat.target_binding_version:
                    return True
        return False

    def taint_of(self, name: str, *, scope: ScopeFlow | None = None) -> Taint:
        if scope is not None:
            return scope.taints.get(name, self._name_taint(name))
        # Prefer module bindings, then a unique function binding.  Ambiguous
        # same-named local bindings are unknown rather than guessed.
        module = self.scopes[0] if self.scopes else None
        if module and name in module.taints:
            return module.taints[name]
        matches = [s.taints[name] for s in self.scopes[1:] if name in s.taints]
        if len(set(matches)) == 1:
            return matches[0]
        return self._name_taint(name)

    def class_of(self, name: str) -> str | None:
        module = self.scopes[0] if self.scopes else None
        if module and name in module.var_classes:
            return module.var_classes[name]
        matches = [s.var_classes[name] for s in self.scopes if name in s.var_classes]
        return matches[0] if len(set(matches)) == 1 else None

    def uses_cv_utility(self) -> bool:
        cv_names = {
            "cross_val_score",
            "cross_validate",
            "cross_val_predict",
            "GridSearchCV",
            "RandomizedSearchCV",
            "HalvingGridSearchCV",
            "HalvingRandomSearchCV",
        }
        return any(call.func_name in cv_names for call in self.calls)

    def evaluation_input_vars(self) -> set[str]:
        out: set[str] = set()
        for split in self.splits:
            out.update(split.input_vars)
        metrics = set(self.adapters.metrics)
        eval_calls = {
            "cross_val_score",
            "cross_validate",
            "cross_val_predict",
            "score",
            "predict",
            "predict_proba",
            "decision_function",
        }
        for call in self.calls:
            if call.func_name in metrics or call.func_name in eval_calls or self.adapters.is_search_constructor(call.func_name):
                out.update(call.arg_vars)
        for fit in self.fit_calls:
            if fit.is_estimator or (fit.class_name and self.adapters.is_search_constructor(fit.class_name)):
                out.update(fit.arg_vars)
        return out

    def fit_output_used_for_evaluation(self, fit: FitCall) -> bool:
        evaluation_bindings: set[tuple[int, str, int]] = set()
        for split in self.splits:
            if split.scope_id != fit.scope_id:
                continue
            evaluation_bindings.update(
                (split.scope_id, name, split.input_binding_versions.get(name, 0))
                for name in split.input_vars
            )
        for call in self.calls:
            if call.scope_id != fit.scope_id:
                continue
            if call.func_name in set(self.adapters.metrics) | {
                "cross_val_score",
                "cross_validate",
                "cross_val_predict",
                "score",
                "predict",
                "predict_proba",
                "decision_function",
            }:
                evaluation_bindings.update(
                    (call.scope_id, name, call.arg_binding_versions.get(name, 0))
                    for name in call.arg_vars
                )
        for candidate in self.fit_calls:
            if candidate.scope_id != fit.scope_id:
                continue
            if candidate.is_estimator or (
                candidate.class_name and self.adapters.is_search_constructor(candidate.class_name)
            ):
                evaluation_bindings.update(
                    (candidate.scope_id, name, candidate.arg_binding_versions.get(name, 0))
                    for name in candidate.arg_vars
                )
        if any(
            (fit.scope_id, name, version) in evaluation_bindings
            for name, version in fit.output_binding_versions.items()
        ):
            return True
        if fit.receiver_var:
            for call in self.calls:
                if (
                    call.scope_id == fit.scope_id
                    and call.func_name
                    in {"transform", "predict", "predict_proba", "decision_function", "score"}
                    and call.receiver_var == fit.receiver_var
                    and call.receiver_binding_version == fit.receiver_binding_version
                ):
                    return True
        return False

    def fit_output_used_by_cv(self, fit: FitCall) -> bool:
        """Return whether a fitted transform feeds a CV/search consumer.

        This is deliberately narrower than :meth:`fit_output_used_for_evaluation`:
        a preprocessing result used for an outer holdout split must not become a
        C001 finding merely because an unrelated CV call appears elsewhere in
        the module.
        """

        cv_names = {
            "cross_val_score",
            "cross_validate",
            "cross_val_predict",
        }
        consumers: list[tuple[int, dict[str, int]]] = [
            (call.scope_id, call.arg_binding_versions)
            for call in self.calls
            if call.func_name in cv_names
        ]
        consumers.extend(
            (search_fit.scope_id, search_fit.arg_binding_versions)
            for search_fit in self.fit_calls
            if search_fit.method == "fit"
            and search_fit.class_name is not None
            and self.adapters.is_search_constructor(search_fit.class_name)
        )
        if fit.output_binding_versions and any(
            fit.scope_id == scope_id
            and any(
                name in bindings
                and bindings.get(name) == version
                for name, version in fit.output_binding_versions.items()
            )
            for scope_id, bindings in consumers
        ):
            return True

        # Follow one supported transform assignment, e.g. ``scaled =
        # scaler.transform(X)``.  The bounded IR intentionally stops here
        # rather than inventing transitive facts through arbitrary functions.
        if fit.receiver_var:
            for call in self.calls:
                if call.func_name not in {"transform", "fit_transform", "fit_resample"}:
                    continue
                if not isinstance(call.node.func, ast.Attribute):
                    continue
                if (
                    call.scope_id != fit.scope_id
                    or _expr_ref(call.node.func.value) != fit.receiver_var
                    or call.receiver_binding_version != fit.receiver_binding_version
                ):
                    continue
                targets = set(self._call_targets.get(id(call.node), []))
                target_versions = call.output_binding_versions
                if targets and any(
                    fit.scope_id == scope_id
                    and any(
                        target in bindings
                        and bindings.get(target) == target_versions.get(target)
                        for target in targets
                    )
                    for scope_id, bindings in consumers
                ):
                    return True

                # Also support an inline transform in a CV argument.
                for cv_call in self.calls:
                    if cv_call.func_name not in cv_names:
                        continue
                    if any(
                        nested is call.node
                        for argument in [*cv_call.node.args, *(kw.value for kw in cv_call.node.keywords)]
                        for nested in ast.walk(argument)
                    ):
                        return True
        return False

    def disconnected_fit_transform_demo(self, fit: FitCall) -> bool:
        return fit.method == "fit_transform" and not self.fit_output_used_for_evaluation(fit)

    def scope_of_line(self, line: int) -> ScopeFlow | None:
        for scope in self.scopes:
            if any(record.line == line for record in scope.fit_calls + scope.calls):
                return scope
        return self.scopes[0] if self.scopes else None

    def _retrofit_pre_split_full_inputs(self) -> None:
        """Snapshot unknown values as full only when a later split proves linkage."""

        for scope in self.scopes:
            if not scope.splits:
                continue
            for fit in scope.fit_calls:
                for split in scope.splits:
                    if split.line <= fit.line:
                        continue
                    connected = any(
                        name in split.input_binding_versions
                        and fit.arg_binding_versions.get(name) == version
                        for name, version in split.input_binding_versions.items()
                    )
                    if not connected:
                        connected = any(
                            name in split.input_binding_versions
                            and fit.output_binding_versions.get(name) == version
                            for name, version in split.input_binding_versions.items()
                        )
                    if not connected:
                        continue
                    fit.arg_taints = [
                        Taint.FULL if taint is Taint.UNKNOWN else taint for taint in fit.arg_taints
                    ]
                    fit.output_taints = [
                        Taint.FULL if taint is Taint.UNKNOWN else taint
                        for taint in fit.output_taints
                    ]
                    break


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for element in target.elts:
            out.extend(_target_names(element))
        return out
    if isinstance(target, ast.Attribute):
        reference = _expr_ref(target)
        return [reference] if reference else []
    if isinstance(target, ast.Subscript):
        reference = _expr_ref(target)
        return [reference] if reference else []
    return []


def _assignment_targets(stmt: ast.AST) -> list[str]:
    if isinstance(stmt, ast.Assign):
        out: list[str] = []
        for target in stmt.targets:
            out.extend(_target_names(target))
        return out
    if isinstance(stmt, ast.AnnAssign):
        return _target_names(stmt.target)
    if isinstance(stmt, ast.NamedExpr):
        return _target_names(stmt.target)
    return []


def _expr_ref(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        parent = _expr_ref(expr.value)
        return f"{parent}.{expr.attr}" if parent else expr.attr
    if isinstance(expr, ast.Subscript):
        parent = _expr_ref(expr.value)
        return f"{parent}[]" if parent else None
    return None


def _expr_names(expr: ast.AST) -> list[str]:
    return [node.id for node in ast.walk(expr) if isinstance(node, ast.Name)]


def _expr_names_from_args(args: list[ast.expr]) -> list[str]:
    out: list[str] = []
    for arg in args:
        out.extend(_expr_names(arg))
    return out


def _constant_int(expr: ast.expr | None) -> int | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _split_output_taints(count: int, input_taint: Taint = Taint.UNKNOWN) -> list[Taint]:
    # train_test_split returns train/test pairs for each positional array.  The
    # return position is stronger evidence than misleading variable names.
    evaluation_role = (
        Taint.VAL
        if input_taint is Taint.TRAIN
        else Taint.TEST
        if input_taint in {Taint.FULL, Taint.UNKNOWN, Taint.TEST}
        else Taint.VAL
    )
    return [Taint.TRAIN if index % 2 == 0 else evaluation_role for index in range(count)]


def _merge_branch_taints(left: dict[str, Taint], right: dict[str, Taint]) -> dict[str, Taint]:
    out: dict[str, Taint] = {}
    for name in set(left) | set(right):
        a, b = left.get(name, Taint.UNKNOWN), right.get(name, Taint.UNKNOWN)
        out[name] = a if a is b else Taint.UNKNOWN
    return out


def _merge_many(branches: list[dict[str, Taint]]) -> dict[str, Taint]:
    if not branches:
        return {}
    out = dict(branches[0])
    for branch in branches[1:]:
        out = _merge_branch_taints(out, branch)
    return out


def _node_key(value: Any) -> tuple[int, int, str]:
    node = getattr(value, "node", None)
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), type(value).__name__)
