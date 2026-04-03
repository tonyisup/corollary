"""Tests for the reasoning layer."""

import pytest

from reasoning.layer import ReasoningLayer, ReasoningResult
from reasoning.provider import ReasoningProvider, ProviderRegistry, MockProvider
from reasoning.mock_layer import MockReasoningLayer


# --- ReasoningResult Tests ---

class TestReasoningResult:
    def test_defaults(self):
        r = ReasoningResult()
        assert r.actions == []
        assert r.updated_nodes == {}
        assert r.confidence == 1.0
        assert r.explanation is None

    def test_with_values(self):
        r = ReasoningResult(
            actions=[{"action": "add_node"}],
            confidence=0.75,
            explanation="test",
        )
        assert len(r.actions) == 1
        assert r.confidence == 0.75
        assert r.explanation == "test"


# --- MockReasoningLayer Tests ---

class TestMockReasoningLayer:
    @pytest.mark.asyncio
    async def test_step_returns_result(self):
        layer = MockReasoningLayer()
        from sag.graph import SemanticAbstractionGraph
        g = SemanticAbstractionGraph()
        result = await layer.step(g, {"utterance": "hello"})
        assert isinstance(result, ReasoningResult)
        assert layer.call_count == 1

    @pytest.mark.asyncio
    async def test_call_count_increments(self):
        layer = MockReasoningLayer()
        from sag.graph import SemanticAbstractionGraph
        g = SemanticAbstractionGraph()
        await layer.step(g, {})
        await layer.step(g, {})
        assert layer.call_count == 2

    @pytest.mark.asyncio
    async def test_with_mock_actions(self):
        actions = [{"action": "add_node", "id": "x"}]
        layer = MockReasoningLayer(actions=actions)
        from sag.graph import SemanticAbstractionGraph
        g = SemanticAbstractionGraph()
        result = await layer.step(g, {})
        assert result.actions == actions


# --- Provider Tests ---

class TestMockProvider:
    def test_name(self):
        p = MockProvider()
        assert p.name() == "mock"

    def test_create_layer(self):
        p = MockProvider()
        layer = p.create_layer()
        assert isinstance(layer, MockReasoningLayer)


class TestProviderRegistry:
    def test_register_and_list(self):
        reg = ProviderRegistry()
        reg.register(MockProvider)
        assert "mock" in reg.list_providers()

    def test_create_layer_via_registry(self):
        reg = ProviderRegistry()
        reg.register(MockProvider)
        layer = reg.create_layer("mock", actions=[{"test": True}])
        assert isinstance(layer, MockReasoningLayer)

    def test_unknown_provider_raises_key_error(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.create_layer("nonexistent")

    def test_duplicate_register_raises(self):
        reg = ProviderRegistry()
        reg.register(MockProvider)
        with pytest.raises(ValueError):
            reg.register(MockProvider)

    def test_has_provider(self):
        reg = ProviderRegistry()
        assert not reg.has("mock")
        reg.register(MockProvider)
        assert reg.has("mock")
