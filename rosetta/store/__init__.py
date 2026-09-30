from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class Hit:
    id: int | str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None


class StoreProtocol(Protocol):
    space_id: str

    def upsert(
        self,
        ids: list[int | str],
        vectors: np.ndarray | list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        vector: np.ndarray | list[float],
        k: int = 10,
        flt: Any | None = None,
    ) -> list[Hit]: ...

    def iterate(self, batch: int = 1024):
        """Yields (ids, vectors, payloads) batches."""
        ...

    def count(self) -> int: ...

    def size_bytes(self) -> int: ...

    def close(self) -> None: ...
