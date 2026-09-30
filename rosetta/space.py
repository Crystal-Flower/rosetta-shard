from dataclasses import dataclass

import numpy as np

from rosetta.adapters.base import l2_normalize
from rosetta.bundle import AdapterBundle


class SpaceMismatchError(Exception):
    """Raised when a query's space_id does not match the target store's space_id."""

    def __init__(self, query_space: str, store_space: str):
        super().__init__(
            f"Space Guard Refusal: Query space '{query_space}' does not match store space '{store_space}'! "
            "Cross-space searches cause silent retrieval collapse."
        )
        self.query_space = query_space
        self.store_space = store_space


@dataclass
class QueryVector:
    """Explicitly typed query vector carrying its resident coordinate space ID."""

    vector: np.ndarray
    space_id: str

    def __post_init__(self):
        self.vector = l2_normalize(np.asarray(self.vector, dtype=np.float32).flatten())


class SpaceGuard:
    """Guards vector stores against cross-space query execution."""

    @staticmethod
    def validate(query_space: str, target_space: str, strict: bool = True) -> None:
        if strict and query_space != target_space:
            raise SpaceMismatchError(query_space=query_space, store_space=target_space)


class AdapterRegistry:
    """
    Directed graph registry of cross-space adapters.
    Finds direct or multi-hop transformation paths between space IDs.
    """

    def __init__(self):
        # (from_space, to_space) -> AdapterBundle
        self.bundles: dict[tuple[str, str], AdapterBundle] = {}

    def register(self, bundle: AdapterBundle) -> None:
        self.bundles[(bundle.from_space, bundle.to_space)] = bundle

    def get_adapter(self, from_space: str, to_space: str) -> AdapterBundle | None:
        return self.bundles.get((from_space, to_space))

    def translate(
        self,
        query: QueryVector,
        target_space: str,
    ) -> QueryVector:
        """Translate a query into the target space using registered adapters."""
        if query.space_id == target_space:
            return query

        # 1. Single hop
        key = (query.space_id, target_space)
        if key in self.bundles:
            bundle = self.bundles[key]
            out_vec = bundle.apply(query.vector)
            return QueryVector(vector=out_vec, space_id=target_space)

        # 2. Multi-hop BFS path search (stretch)
        visited = {query.space_id}
        queue = [[(query.space_id, None)]]
        target_path = None

        while queue:
            path = queue.pop(0)
            curr = path[-1][0]
            if curr == target_space:
                target_path = path
                break

            for (f, t), b in self.bundles.items():
                if f == curr and t not in visited:
                    visited.add(t)
                    queue.append(path + [(t, b)])

        if target_path:
            curr_vec = query.vector
            for t_space, bundle in target_path[1:]:
                curr_vec = bundle.apply(curr_vec)
            return QueryVector(vector=curr_vec, space_id=target_space)

        raise ValueError(
            f"No adapter path found in registry from space '{query.space_id}' to '{target_space}'"
        )
