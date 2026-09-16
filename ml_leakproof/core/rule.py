"""Rule base classes for the three layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .models import Category, Finding, Layer, Severity

if TYPE_CHECKING:
    from ..runtime.taint import RuntimeEvent
    from .context import DataContext, RuntimeContext, StaticContext


class Rule(ABC):
    """Base class shared by all rules.

    Subclasses set the class attributes below and implement exactly one of the
    layer-specific ``check``/``on_event`` hooks (via the layer mix-ins).
    """

    id: str  # "P001"
    name: str  # "fit on full dataset before split"
    category: Category
    severity: Severity  # default; overridable via config
    layers: tuple[Layer, ...]
    references: tuple[str, ...] = ()
    enabled_by_default: bool = True
    # True means every finding from this rule is advisory and must never gate.
    advisory_only: bool = False
    # short rationale shown by `ml_leakproof explain`
    rationale: str = ""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} {self.id}>"


class StaticRule(Rule):
    @abstractmethod
    def check(self, ctx: StaticContext) -> Iterable[Finding]: ...


class DataRule(Rule):
    @abstractmethod
    def check(self, ctx: DataContext) -> Iterable[Finding]: ...


class RuntimeRule(Rule):
    @abstractmethod
    def on_event(
        self, event: RuntimeEvent, ctx: RuntimeContext
    ) -> Iterable[Finding]: ...
