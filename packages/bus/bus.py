"""In-process pub/sub bus for Corollary event routing.

TODO: Replace with gRPC for cross-process modules.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable, Type

from .events import BusEvent

Handler = Callable[[BusEvent], None | Callable[[], None]]


class MessageBus:
    """Simple async event bus with typed subscriptions."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[BusEvent]]] = defaultdict(list)
        self._global: list[asyncio.Queue[BusEvent]] = []

    # ── subscribe ─────────────────────────────────────────────────────────

    def subscribe(self, queue: asyncio.Queue[BusEvent], event_types: list[str] | None = None) -> None:
        """Attach a queue to receive events of given types, or all if None."""
        if event_types is None:
            self._global.append(queue)
        else:
            for t in event_types:
                self._queues[t].append(queue)

    # ── publish ───────────────────────────────────────────────────────────

    async def publish(self, event: BusEvent) -> None:
        """Fan-out an event to all subscribed queues."""
        t = event.type
        targets = list(self._queues.get(t, [])) + list(self._global)
        for q in targets:
            await q.put(event)
