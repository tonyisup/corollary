"""Mock reasoning layer for testing.

Provides deterministic, predictable behavior without external dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from reasoning.layer import ReasoningLayer, ReasoningResult

if TYPE_CHECKING:
    from sag.graph import SemanticAbstractionGraph


class MockReasoningLayer(ReasoningLayer):
    """A no-op reasoning layer for testing infrastructure.

    Returns empty results by default. Configure with `actions` to
    return specific outputs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mock_actions = kwargs.get("actions", [])
        self._last_context: Dict[str, Any] = {}
        self._call_count = 0

    async def step(
        self,
        graph: SemanticAbstractionGraph,
        context: Dict[str, Any],
    ) -> ReasoningResult:
        self._last_context = context
        self._call_count += 1
        return ReasoningResult(
            actions=list(self._mock_actions),
            confidence=1.0,
            explanation=f"Mock reasoning step #{self._call_count}",
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_context(self) -> Dict[str, Any]:
        """Return the context from the most recent step() call."""
        return dict(self._last_context)
