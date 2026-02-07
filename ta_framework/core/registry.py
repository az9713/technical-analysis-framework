"""Decorator-based plugin registry for indicators, strategies, and data providers."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Generic plugin registry.

    Usage:
        indicator_registry = Registry("indicators")

        @indicator_registry.register("sma")
        class SMAIndicator:
            ...

        cls = indicator_registry.get("sma")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, Any] = {}

    def register(self, key: str, **meta: Any) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self._store[key] = {"obj": obj, "meta": meta}
            return obj
        return decorator

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            raise KeyError(f"'{key}' not found in {self.name} registry. Available: {self.keys()}")
        return entry["obj"]

    def get_meta(self, key: str) -> dict[str, Any]:
        entry = self._store.get(key)
        if entry is None:
            raise KeyError(f"'{key}' not found in {self.name} registry.")
        return entry["meta"]

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def items(self) -> list[tuple[str, Any]]:
        return [(k, v["obj"]) for k, v in self._store.items()]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)


# Global registries
indicator_registry = Registry("indicators")
strategy_registry = Registry("strategies")
provider_registry = Registry("providers")
