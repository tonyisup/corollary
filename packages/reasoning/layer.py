"""Abstract reasoning layer interface.

The reasoning layer drives cognitive processing — interpreting input,
updating the SAG, and deciding what to do next.

Provider-agnostic: any reasoning backend (LLM, rules, hybrid) plugs in
via the ProviderRegistry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sag.graph import SemanticAbstractionGraph


@dataclass
class ReasoningResult:
    """Output of a reasoning step."""

    actions: List[Dict[str, Any]] = field(default_factory=list)
    """Concrete actions to apply (e.g. add_node, merge_nodes)."""

    updated_nodes: Dict[str, Any] = field(default_factory=dict)
    """Node deltas keyed by node_id."""

    confidence: float = 1.0
    """Overall confidence in this reasoning result."""

    explanation: Optional[str] = None
    """Human-readable rationale for the decision."""


class ReasoningLayer(ABC):
    """Abstract base for all reasoning implementations.

    Implementations MUST override `step()`. They MAY override
    `bulk_step()` for batch processing efficiency.

    The layer receives the current SAG state + context and returns
    a ReasoningResult describing what changes to make.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the reasoning layer.

        Subclasses should accept provider-specific kwargs
        and forward unknown ones to super().__init__().
        """
        self._config: Dict[str, Any] = kwargs

    @abstractmethod
    async def step(
        self,
        graph: SemanticAbstractionGraph,
        context: Dict[str, Any],
    ) -> ReasoningResult:
        """Perform one reasoning step.

        Args:
            graph: Current SAG state (read-only from reasoning's perspective).
            context: Session data — recent utterances, salience scores, etc.

        Returns:
            A ReasoningResult describing suggested graph changes.
        """
        ...

    async def bulk_step(
        self,
        graph: SemanticAbstractionGraph,
        contexts: List[Dict[str, Any]],
    ) -> List[ReasoningResult]:
        """Process multiple contexts in one reasoning step.

        Default implementation calls step() sequentially.
        Override for parallel processing.
        """
        results: List[ReasoningResult] = []
        for ctx in contexts:
            result = await self.step(graph, ctx)
            results.append(result)
        return results

    @property
    def config(self) -> Dict[str, Any]:
        """Return the layer's configuration."""
        return dict(self._config)
