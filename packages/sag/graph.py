"""SemanticAbstractionGraph — the PFC's working memory backbone."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sag.models import (
    ChangeKind,
    Edge,
    EdgeAssertion,
    GraphChange,
    Node,
)


class SemanticAbstractionGraph:
    """A directed, labeled graph for semantic concepts.

    Nodes are semantic concepts with multi-resolution descriptions.
    Edges are typed relationships with per-module assertions and consensus confidence.

    Thread safety: NOT thread-safe. External synchronization required
    if accessed from multiple coroutines.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        # Edge key: (source_id, target_id, type)
        self._edges: dict[tuple[str, str, str], Edge] = {}
        # For resolve(): token -> list of candidate node ids
        self._token_index: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Core mutations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add or replace a node."""
        self._nodes[node.id] = node
        # Index grounding anchors for resolve()
        for anchor in node.grounding:
            key = anchor.lower()
            if node.id not in self._token_index[key]:
                self._token_index[key].append(node.id)

    def add_edge(self, edge: Edge) -> None:
        """Add or replace an edge."""
        key = (edge.source, edge.target, edge.type)
        self._edges[key] = edge

    def add_assertion(
        self,
        source: str,
        target: str,
        rel_type: str,
        assertion: EdgeAssertion,
    ) -> Edge:
        """Add a module's assertion to an edge, updating consensus confidence.

        Creates the edge if it doesn't exist.
        """
        key = (source, target, rel_type)
        if key in self._edges:
            existing = self._edges[key]
            # Check if this module already asserted
            new_assertions = [
                a for a in existing.assertions if a.asserted_by != assertion.asserted_by
            ]
            new_assertions.append(assertion)
        else:
            new_assertions = [assertion]

        # Consensus = mean of assertion confidences
        consensus = sum(a.confidence for a in new_assertions) / len(new_assertions)

        self._edges[key] = Edge(
            source=source,
            target=target,
            type=rel_type,
            confidence=consensus,
            assertions=new_assertions,
        )
        return self._edges[key]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        """Return a node by id, or None."""
        return self._nodes.get(node_id)

    def get_edge(
        self, source: str, target: str, rel_type: str
    ) -> Edge | None:
        """Return an edge, or None."""
        return self._edges.get((source, target, rel_type))

    def nodes(self) -> list[Node]:
        """Return all nodes."""
        return list(self._nodes.values())

    def edges(self) -> list[Edge]:
        """Return all edges."""
        return list(self._edges.values())

    def neighbors(
        self, node_id: str, rel_types: list[str] | None = None
    ) -> list[Edge]:
        """Return outgoing edges for a node, optionally filtered by type."""
        result = []
        for (src, tgt, _type), edge in self._edges.items():
            if src != node_id:
                continue
            if rel_types is not None and _type not in rel_types:
                continue
            result.append(edge)
        return result

    # ------------------------------------------------------------------
    # High-level operations (from sag.md API surface)
    # ------------------------------------------------------------------

    def resolve(self, token: str, context: dict | None = None) -> Node | None:
        """Resolve a token (word/phrase) to the best-matching node given context.

        Uses grounding anchors for initial matching, then context filtering.
        Returns None if no match is found.

        M0 implementation: simple grounding anchor lookup with optional
        context key/value filtering. Future: embeddings-based candidate search.
        """
        candidates = self._token_index.get(token.lower(), [])
        if not candidates:
            return None

        if not context:
            return self._nodes[candidates[0]]

        # Filter candidates by context overlap
        best_score = -1
        best_node: Node | None = None
        for cid in candidates:
            node = self._nodes[cid]
            score = 0
            for k, v in context.items():
                if k in node.context and node.context[k] == v:
                    score += 2
                elif k in node.context:
                    score += 1  # key match but different value
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def traverse(
        self,
        start: str,
        rel_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[Node]:
        """Traverse the graph from a starting node, collecting reached nodes.

        Returns nodes reachable within `depth` steps, respecting
        optional `rel_types` filter.
        """
        if start not in self._nodes:
            return []

        visited: set[str] = set()
        frontier: list[str] = [start]
        result: list[Node] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                for edge in self.neighbors(node_id, rel_types):
                    if edge.target not in visited:
                        next_frontier.append(edge.target)
                        target_node = self._nodes.get(edge.target)
                        if target_node is not None and target_node not in result:
                            result.append(target_node)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def salience(self, node_id: str) -> float:
        """Compute a salience score for a node.

        M0: sum of incoming edge confidences.
        More connected + higher-confidence = more salient.
        Future: incorporate traversal frequency, recency, novelty.
        """
        if node_id not in self._nodes:
            return 0.0

        score = 0.0
        for (_src, _tgt, _type), edge in self._edges.items():
            if _tgt == node_id:
                score += edge.confidence
        return score

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def diff(self, other: "SemanticAbstractionGraph") -> list[GraphChange]:
        """Compare this graph against another and return changes.

        Useful for tracking graph evolution over time or between snapshots.
        """
        changes: list[GraphChange] = []

        # Node diffs
        all_node_ids = set(self._nodes.keys()) | set(other._nodes.keys())
        for nid in all_node_ids:
            before = self._nodes.get(nid)
            after = other._nodes.get(nid)
            if before is None:
                changes.append(
                    GraphChange(kind=ChangeKind.NODE_ADDED, node_id=nid, after=after)
                )
            elif after is None:
                changes.append(
                    GraphChange(kind=ChangeKind.NODE_REMOVED, node_id=nid, before=before)
                )
            elif before != after:
                changes.append(
                    GraphChange(
                        kind=ChangeKind.NODE_MODIFIED,
                        node_id=nid,
                        before=before,
                        after=after,
                    )
                )

        # Edge diffs
        all_edge_keys = set(self._edges.keys()) | set(other._edges.keys())
        for ekey in all_edge_keys:
            before = self._edges.get(ekey)
            after = other._edges.get(ekey)
            if before is None:
                changes.append(
                    GraphChange(kind=ChangeKind.EDGE_ADDED, edge_key=ekey, after=after)
                )
            elif after is None:
                changes.append(
                    GraphChange(kind=ChangeKind.EDGE_REMOVED, edge_key=ekey, before=before)
                )
            elif before != after:
                changes.append(
                    GraphChange(
                        kind=ChangeKind.EDGE_MODIFIED,
                        edge_key=ekey,
                        before=before,
                        after=after,
                    )
                )

        return changes

    def snapshot(self) -> "SemanticAbstractionGraph":
        """Create a shallow copy of the current graph state for diffing."""
        copy = SemanticAbstractionGraph()
        for node in self._nodes.values():
            copy.add_node(node)
        for edge in self._edges.values():
            copy.add_edge(edge)
        return copy
