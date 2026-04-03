# Semantic Abstraction Graph (SAG) — Corollary Design Doc

> **Status:** Design (realized in Corollary)  
> **Created:** 2026-04-03  
> **Motivation:** Give the PFC a native, grounded knowledge representation to prevent internal semantic drift.

## Why SAG Exists

Corollary's internal modules (PFC ↔ Hippocampus ↔ Temporal Semantic System) exchange representations of concepts, entities, and relationships. Without explicit semantic grounding, each module develops its own implicit meaning for the same tokens — classic semantic drift. The SAG is a structured graph that provides a **shared, stable vocabulary** for all internal communication.

This is not a "training goal" bolted on later. It is infrastructure from M0.

## What It Is

A directed, labeled graph where:

- **Nodes** = semantic concepts (entities, states, abstractions)
- **Edges** = typed relationships (is-a, part-of, causes, contradicts, implies, etc.)
- **Node payloads** = multi-resolution descriptions:
  - `core`: stable, high-level definition (rarely changes)
  - `context`: situational parameters that shift per conversation
  - `grounding`: observable anchors — words, sounds, sensor readings that map to this node
- **Edges** carry:
  - `type`: relationship taxonomy
  - `confidence`: float [0, 1] updated through interaction
  - `source`: which module asserted this (PFC, STG, etc.)

## Role in Corollary

### M0–M2: PFC Working Memory Backbone

The SAG replaces freeform context strings in the PFC. When the dialogue controller needs to reason about something, it queries the graph rather than relying on raw text history. Concrete effects:

- **Disambiguation at M0:** When ASR produces ambiguous tokens, the PFC resolves them against the graph's context state before forwarding to dialogue logic.
- **Efference copy alignment (cerebellum ↔ SAG):** The cerebellum's forward model doesn't just predict spectrograms — it predicts *meaning*. "I said X" in the cerebellum maps to a node in the SAG, and the cerebellum can verify: did the STG hear back the node I intended?
- **Memory write salience:** The salience network uses SAG traversal patterns to detect novelty. A concept visited frequently through new paths = high salience. Write to episodic memory.

### M3–M4: Self/World Attribution

The SAG tracks speaker attribution at the semantic level. Not just "voice A said this" but "voice A asserted relationship between concept X and concept Y." This lets the self-voice detector use meaning coherence as a signal — if the STG attributed a frame to "world" but the semantic content matches a node the PFC is actively holding, that's a candidate for self-leakage.

### M5: Introspection

Self-model = graph of graph. The SAG accumulates a "self-concept" subtree: nodes like `Corollary.agent_voice`, `Corollary.speech_patterns`, `Corollary.known_subjects`. The self-model in M5 is SAG nodes with edges pointing back to the graph itself — introspection as graph traversal of self-referential structure.

## Relationship to the Semantics Project

| | Semantics Project | SAG in Corollary |
|---|---|---|
| **Scope** | General theory of shared meaning between agents | Specific implementation as PFC's knowledge store |
| **Goal** | Prevent drift across agent-human communication | Prevent drift across internal modules |
| **Output** | Methodology, probing techniques, formalisms | Working code: `packages/sag/` |

Semantics research is the *theory*. SAG in Corollary is the *instantiation*. They inform each other: Corollary provides real usage data; semantics provides design principles.

## Milestones

| Milestone | SAG Deliverable |
|-----------|-----------------|
| M0 | Graph data structure + Pydantic schemas for nodes/edges. PFC uses it for basic entity tracking in simulated loop. |
| M1 | Populate graph from real conversation turns. Multi-speaker attribution on edges. |
| M2 | SAG-informed disambiguation: PFC uses graph context to resolve ASR ambiguity. |
| M3 | Self-concept subtree begins forming. Graph detects when self-predicted meaning ≠ heard meaning. |
| M4 | Salience-driven graph traversal informs memory writes. Cross-concept edge learning from dialogue. |
| M5 | Introspective subtree: agent can query its own knowledge state and report confidence levels. |

## API Surface

```python
# packages/sag/__init__.py
class Node:
    id: str
    core: str          # stable definition
    context: dict      # situational params
    grounding: list[str] # observable anchors

class Edge:
    source: str        # node id
    target: str
    type: str
    confidence: float
    asserted_by: str   # module name

class SemanticAbstractionGraph:
    add_node(node: Node) -> None
    add_edge(edge: Edge) -> None
    resolve(token: str, context: dict) -> Node | None
    traverse(start: str, rel_types: list[str], depth: int) -> list[Node]
    salience(node_id: str) -> float  # for salience network
    diff(before: "SAG", after: "SAG") -> list[Change]  # for introspection
```

## Design Decisions

- **Graph first, embeddings second:** Vector similarity (FAISS/Chroma) is complementary but lossy. The SAG preserves explicit structure. Use embeddings to *find candidate nodes*, use the graph to *confirm relationships*.
- **Mutable edges, stable cores:** Node cores rarely change. Edge confidences update continuously. This mirrors how human concepts are stable but our beliefs about their relationships shift.
- **Multi-source assertions:** Multiple modules can assert the same edge. The graph tracks assertions and derives consensus. Disagreement *between modules* is itself a useful signal for salience.
