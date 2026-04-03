"""Tests for the Semantic Abstraction Graph."""

import pytest

from sag.models import Node, Edge, EdgeAssertion, EdgeType, ChangeKind
from sag.graph import SemanticAbstractionGraph


# --- Fixtures ---

@pytest.fixture
def empty_graph() -> SemanticAbstractionGraph:
    return SemanticAbstractionGraph()


@pytest.fixture
def sample_graph() -> SemanticAbstractionGraph:
    g = SemanticAbstractionGraph()
    g.add_node(Node(id="cat", core="A small domesticated feline",
                    context={"domain": "animals"}, grounding=["cat", "kitty", "meow"]))
    g.add_node(Node(id="mammal", core="A warm-blooded vertebrate with hair",
                    context={"domain": "taxonomy"}, grounding=["mammal"]))
    g.add_node(Node(id="pet", core="A companion animal kept for enjoyment",
                    context={"domain": "lifestyle"}, grounding=["pet", "companion"]))
    g.add_edge(Edge(source="cat", target="mammal", type="is-a", confidence=0.95))
    g.add_edge(Edge(source="cat", target="pet", type="related-to", confidence=0.8))
    return g


# --- Node Tests ---

class TestNode:
    def test_minimal_node(self):
        n = Node(id="test", core="A test concept")
        assert n.id == "test"
        assert n.core == "A test concept"
        assert n.context == {}
        assert n.grounding == []

    def test_full_node(self):
        n = Node(id="test", core="A test concept",
                 context={"key": "val"}, grounding=["anchor1", "anchor2"])
        assert n.context == {"key": "val"}
        assert n.grounding == ["anchor1", "anchor2"]

    def test_node_is_frozen(self):
        n = Node(id="test", core="Immutable")
        with pytest.raises(Exception):  # pydantic ValidationError
            n.core = "changed"  # type: ignore[misc]


# --- Edge Tests ---

class TestEdge:
    def test_edge_with_assertions(self):
        a = EdgeAssertion(asserted_by="pfc", confidence=0.9)
        e = Edge(source="a", target="b", type="is-a",
                confidence=0.9, assertions=[a])
        assert e.source == "a"
        assert e.target == "b"
        assert e.type == "is-a"
        assert len(e.assertions) == 1

    def test_edge_is_frozen(self):
        e = Edge(source="a", target="b", type="is-a", confidence=0.5)
        with pytest.raises(Exception):
            e.confidence = 1.0  # type: ignore[misc]

    def test_edge_types_enum(self):
        assert EdgeType.IS_A.value == "is-a"
        assert EdgeType.CAUSES.value == "causes"
        assert EdgeType.IMPLIES.value == "implies"


# --- Graph: Mutations ---

class TestGraphAddNode:
    def test_add_and_retrieve(self, empty_graph):
        n = Node(id="x", core="Concept X")
        empty_graph.add_node(n)
        assert empty_graph.get_node("x") is n

    def test_get_nonexistent(self, empty_graph):
        assert empty_graph.get_node("nope") is None

    def test_replace_node(self, empty_graph):
        n1 = Node(id="x", core="Old")
        n2 = Node(id="x", core="New")
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        assert empty_graph.get_node("x").core == "New"  # type: ignore[union-attr]

    def test_nodes_returns_all(self, empty_graph):
        empty_graph.add_node(Node(id="a", core="A"))
        empty_graph.add_node(Node(id="b", core="B"))
        ids = [n.id for n in empty_graph.nodes()]
        assert set(ids) == {"a", "b"}


class TestGraphAddEdge:
    def test_add_and_retrieve(self, empty_graph):
        e = Edge(source="x", target="y", type="is-a", confidence=0.9)
        empty_graph.add_edge(e)
        result = empty_graph.get_edge("x", "y", "is-a")
        assert result is e

    def test_get_nonexistent_edge(self, empty_graph):
        assert empty_graph.get_edge("x", "y", "is-a") is None

    def test_edges_returns_all(self, empty_graph):
        empty_graph.add_edge(Edge(source="a", target="b", type="r", confidence=0.5))
        empty_graph.add_edge(Edge(source="b", target="c", type="s", confidence=0.6))
        assert len(empty_graph.edges()) == 2


class TestGraphAddAssertion:
    def test_first_assertion_creates_edge(self, empty_graph):
        a = EdgeAssertion(asserted_by="pfc", confidence=0.8)
        e = empty_graph.add_assertion("x", "y", "causes", a)
        assert e.confidence == 0.8
        assert len(e.assertions) == 1

    def test_consensus_mean(self, empty_graph):
        empty_graph.add_assertion("x", "y", "causes",
                                  EdgeAssertion(asserted_by="pfc", confidence=0.9))
        empty_graph.add_assertion("x", "y", "causes",
                                  EdgeAssertion(asserted_by="stg", confidence=0.5))
        e = empty_graph.get_edge("x", "y", "causes")
        assert e is not None
        assert len(e.assertions) == 2
        assert e.confidence == pytest.approx(0.7)  # (0.9 + 0.5) / 2

    def test_replace_existing_module_assertion(self, empty_graph):
        empty_graph.add_assertion("x", "y", "is-a",
                                  EdgeAssertion(asserted_by="pfc", confidence=0.8))
        empty_graph.add_assertion("x", "y", "is-a",
                                  EdgeAssertion(asserted_by="pfc", confidence=0.95))  # update
        e = empty_graph.get_edge("x", "y", "is-a")
        assert e is not None
        assert len(e.assertions) == 1  # same module, replaced
        assert e.assertions[0].confidence == 0.95


# --- Graph: Neighbors ---

class TestGraphNeighbors:
    def test_all_neighbors(self, sample_graph):
        edges = sample_graph.neighbors("cat")
        assert len(edges) == 2

    def test_filter_by_type(self, sample_graph):
        edges = sample_graph.neighbors("cat", rel_types=["is-a"])
        assert len(edges) == 1
        assert edges[0].target == "mammal"

    def test_nonexistent_node(self, sample_graph):
        assert sample_graph.neighbors("ghost") == []


# --- Graph: Resolve ---

class TestGraphResolve:
    def test_resolve_by_grounding(self, sample_graph):
        node = sample_graph.resolve("cat")
        assert node is not None
        assert node.id == "cat"

    def test_resolve_case_insensitive(self, sample_graph):
        node = sample_graph.resolve("CAT")
        assert node is not None
        assert node.id == "cat"

    def test_resolve_with_context_match(self):
        g = SemanticAbstractionGraph()
        g.add_node(Node(id="bank_river", core="River bank",
                        context={"domain": "geography"}, grounding=["bank"]))
        g.add_node(Node(id="bank_money", core="Financial institution",
                        context={"domain": "finance"}, grounding=["bank"]))
        node = g.resolve("bank", context={"domain": "finance"})
        assert node is not None
        assert node.id == "bank_money"

    def test_resolve_no_match(self, sample_graph):
        assert sample_graph.resolve("zebra") is None


# --- Graph: Traverse ---

class TestGraphTraverse:
    def test_depth_1(self, sample_graph):
        nodes = sample_graph.traverse("cat", depth=1)
        ids = {n.id for n in nodes}
        assert ids == {"mammal", "pet"}

    def test_depth_2(self, sample_graph):
        sample_graph.add_node(Node(id="animal", core="A living organism",
                                    context={"domain": "taxonomy"}, grounding=["animal"]))
        sample_graph.add_edge(Edge(source="mammal", target="animal", type="is-a",
                                    confidence=0.99))
        nodes = sample_graph.traverse("cat", depth=2)
        ids = {n.id for n in nodes}
        assert ids == {"mammal", "pet", "animal"}

    def test_filter_by_rel_type(self, sample_graph):
        nodes = sample_graph.traverse("cat", rel_types=["is-a"], depth=1)
        ids = {n.id for n in nodes}
        assert ids == {"mammal"}

    def test_nonexistent_start(self, sample_graph):
        assert sample_graph.traverse("ghost") == []

    def test_no_outgoing_edges(self):
        g = SemanticAbstractionGraph()
        g.add_node(Node(id="alone", core="No edges"))
        assert g.traverse("alone", depth=3) == []

    def test_cycle_detection(self):
        g = SemanticAbstractionGraph()
        g.add_node(Node(id="a", core="A"))
        g.add_node(Node(id="b", core="B"))
        g.add_edge(Edge(source="a", target="b", type="r", confidence=0.5))
        g.add_edge(Edge(source="b", target="a", type="r", confidence=0.5))
        result = g.traverse("a", depth=5)
        assert len(result) == 1
        assert result[0].id == "b"


# --- Graph: Salience ---

class TestGraphSalience:
    def test_salience_from_incoming(self, sample_graph):
        s = sample_graph.salience("mammal")
        assert s == pytest.approx(0.95)

    def test_salience_no_incoming(self, sample_graph):
        assert sample_graph.salience("cat") == 0.0

    def test_salience_nonexistent(self, sample_graph):
        assert sample_graph.salience("ghost") == 0.0

    def test_multiple_incoming(self):
        g = SemanticAbstractionGraph()
        g.add_node(Node(id="hub", core="Central concept"))
        g.add_node(Node(id="a", core="A"))
        g.add_node(Node(id="b", core="B"))
        g.add_edge(Edge(source="a", target="hub", type="r", confidence=0.3))
        g.add_edge(Edge(source="b", target="hub", type="r", confidence=0.7))
        assert g.salience("hub") == pytest.approx(1.0)


# --- Graph: Diff ---

class TestGraphDiff:
    def test_added_node(self):
        g1 = SemanticAbstractionGraph()
        g2 = SemanticAbstractionGraph()
        g2.add_node(Node(id="x", core="New"))
        changes = g1.diff(g2)
        assert any(c.kind == ChangeKind.NODE_ADDED and c.node_id == "x" for c in changes)

    def test_removed_node(self):
        g1 = SemanticAbstractionGraph()
        g1.add_node(Node(id="x", core="Old"))
        g2 = SemanticAbstractionGraph()
        changes = g1.diff(g2)
        assert any(c.kind == ChangeKind.NODE_REMOVED and c.node_id == "x" for c in changes)

    def test_modified_node(self):
        g1 = SemanticAbstractionGraph()
        g1.add_node(Node(id="x", core="v1"))
        g2 = SemanticAbstractionGraph()
        g2.add_node(Node(id="x", core="v2"))
        changes = g1.diff(g2)
        assert any(c.kind == ChangeKind.NODE_MODIFIED and c.node_id == "x" for c in changes)

    def test_added_edge(self):
        g1 = SemanticAbstractionGraph()
        g2 = SemanticAbstractionGraph()
        g2.add_edge(Edge(source="a", target="b", type="r", confidence=0.5))
        changes = g1.diff(g2)
        assert any(c.kind == ChangeKind.EDGE_ADDED for c in changes)

    def test_empty_diff(self):
        g1 = SemanticAbstractionGraph()
        g2 = SemanticAbstractionGraph()
        assert g1.diff(g2) == []

    def test_snapshot_and_diff(self, sample_graph):
        snap = sample_graph.snapshot()
        sample_graph.add_node(Node(id="new_thing", core="Something new"))
        changes = snap.diff(sample_graph)
        assert any(c.kind == ChangeKind.NODE_ADDED and c.node_id == "new_thing" for c in changes)
