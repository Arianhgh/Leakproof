"""Public extension API surface for third-party plugins.

A plugin is a normal pip package that declares entry points::

    [project.entry-points."ml_leakproof.rules"]
    myorg = "myorg_leakproof:rules"

    [project.entry-points."ml_leakproof.adapters"]
    myorg = "myorg_leakproof:adapter"

where ``rules`` is a callable returning a list of rule instances and ``adapter``
returns a FrameworkAdapter. Plugin rule IDs MUST be vendor-prefixed
(e.g. ``myorg-X001``) to avoid collisions with built-ins.

See docs/plugins.md for a worked example.
"""

from __future__ import annotations

from .core.models import Category, Finding, Fix, Layer, Location, Severity
from .core.registry import register
from .core.rule import DataRule, Rule, RuntimeRule, StaticRule
from .static.adapters import FrameworkAdapter, register_adapter

__all__ = [
    "register",
    "register_adapter",
    "Rule",
    "StaticRule",
    "DataRule",
    "RuntimeRule",
    "FrameworkAdapter",
    "Finding",
    "Fix",
    "Severity",
    "Category",
    "Layer",
    "Location",
]
