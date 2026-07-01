"""Built-in static rules.

Importing this package registers every static rule via the ``@register``
decorator applied in the submodules.
"""

from __future__ import annotations

from . import (  # noqa: F401
    adaptivity,
    cv,
    determinism,
    metric,
    preprocessing,
    split,
    temporal,
)
