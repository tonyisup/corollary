"""Semantic Abstraction Graph — Corollary's internal knowledge representation.

The SAG is a directed, labeled graph where nodes represent semantic concepts
(entities, states, abstractions) and edges represent typed relationships.
It serves as the PFC's working memory backbone and a shared vocabulary
across all internal modules.
"""

from sag.models import Node, Edge, EdgeAssertion, GraphChange
from sag.graph import SemanticAbstractionGraph

__all__ = [
    "SemanticAbstractionGraph",
    "Node",
    "Edge",
    "EdgeAssertion",
    "GraphChange",
]
