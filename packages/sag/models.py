"""Pydantic models for the Semantic Abstraction Graph."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class Node(BaseModel):
    """A semantic concept — entity, state, or abstraction."""

    id: str
    core: str = Field(description="Stable, high-level definition; rarely changes")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Situational parameters that shift per conversation",
    )
    grounding: list[str] = Field(
        default_factory=list,
        description="Observable anchors — words, sounds, sensor readings that map to this node",
    )

    model_config = {"frozen": True}


class EdgeType(str, Enum):
    """Standard relationship types between nodes."""

    IS_A = "is-a"
    PART_OF = "part-of"
    CAUSES = "causes"
    CONTRADICTS = "contradicts"
    IMPLIES = "implies"
    RELATED_TO = "related-to"
    SIMILAR_TO = "similar-to"
    OPPOSITE_OF = "opposite-of"
    USED_FOR = "used-for"
    HAS_PROPERTY = "has-property"


class EdgeAssertion(BaseModel):
    """A single module's assertion of a relationship.

    Multiple modules can assert the same edge; the graph tracks
    assertions and derives consensus.
    """

    asserted_by: str = Field(description="Module name that asserted this edge")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this specific assertion"
    )


class Edge(BaseModel):
    """A typed relationship between two nodes."""

    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    type: str = Field(description="Relationship type (EdgeType or custom)")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Derived confidence from all assertions (consensus)",
    )
    assertions: list[EdgeAssertion] = Field(
        default_factory=list,
        description="Individual module assertions that make up this edge",
    )

    model_config = {"frozen": True}


class ChangeKind(str, Enum):
    """Type of change detected in a graph diff."""

    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_MODIFIED = "node_modified"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    EDGE_MODIFIED = "edge_modified"


class GraphChange(BaseModel):
    """A single change between two graph snapshots."""

    kind: ChangeKind
    node_id: Optional[str] = None
    edge_key: Optional[Tuple[str, str, str]] = None  # (source, target, type)
    before: Optional[Any] = None
    after: Optional[Any] = None
