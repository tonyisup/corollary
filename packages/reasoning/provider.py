"""Provider registry for reasoning backends.

A ReasoningProvider is a callable (or class) that supplies the
reasoning capabilities for a ReasoningLayer. The registry allows
multiple providers to be configured and swapped at runtime.

Provider types might include:
  - "openai"   — GPT-based reasoning
  - "anthropic" — Claude-based reasoning
  - "local"    — Local model (Ollama, LM Studio)
  - "mock"     — Deterministic stubs for testing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

if TYPE_CHECKING:
    from reasoning.layer import ReasoningLayer


class ReasoningProvider(ABC):
    """Abstract base for reasoning provider implementations.

    A provider knows how to instantiate and configure a specific
    ReasoningLayer backend.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique provider name (e.g. 'openai', 'mock')."""
        ...

    @abstractmethod
    def create_layer(self, **kwargs: Any) -> "ReasoningLayer":
        """Create and return a configured ReasoningLayer instance.

        Args:
            **kwargs: Provider-specific configuration options.

        Returns:
            A configured ReasoningLayer ready for use.
        """
        ...


class ProviderRegistry:
    """Registry that maps provider names to provider classes.

    Usage:
        registry = ProviderRegistry()
        registry.register(MockProvider)
        layer = registry.create_layer("mock", **config)
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Type[ReasoningProvider]] = {}

    def register(self, provider_cls: Type[ReasoningProvider]) -> None:
        """Register a provider class, auto-discovering its name.

        Args:
            provider_cls: A ReasoningProvider subclass (must be instantiable with no args).

        Raises:
            ValueError: If a provider with the same name is already registered.
        """
        instance = provider_cls()
        name = instance.name()
        if name in self._providers:
            raise ValueError(
                f"Provider '{name}' is already registered; "
                f"cannot override with {provider_cls.__name__}"
            )
        self._providers[name] = provider_cls

    def create_layer(
        self, provider_name: str, **kwargs: Any
    ) -> "ReasoningLayer":
        """Create a ReasoningLayer using the named provider.

        Args:
            provider_name: The registered provider name.
            **kwargs: Configuration to pass to the provider's create_layer().

        Returns:
            A configured ReasoningLayer instance.

        Raises:
            KeyError: If the provider name is not registered.
        """
        if provider_name not in self._providers:
            raise KeyError(
                f"Provider '{provider_name}' not found. "
                f"Available: {list(self._providers.keys())}"
            )
        provider: ReasoningProvider = self._providers[provider_name]()
        return provider.create_layer(**kwargs)

    def list_providers(self) -> list:
        """Return a list of registered provider names."""
        return list(self._providers.keys())

    def has(self, provider_name: str) -> bool:
        """Check if a provider name is registered."""
        return provider_name in self._providers


class MockProvider(ReasoningProvider):
    """A stub provider for testing. Creates a MockReasoningLayer."""

    def name(self) -> str:
        return "mock"

    def create_layer(self, **kwargs: Any) -> "ReasoningLayer":
        from reasoning.mock_layer import MockReasoningLayer
        return MockReasoningLayer(**kwargs)
