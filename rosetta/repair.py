import numpy as np

from rosetta.store.edge_store import EdgeStore


class HotSetRepair:
    """
    Byte-Budgeted Hot-Set Repair Manager.
    Selects high-impact documents whose translated vectors have high error
    and frequent user queries, replacing them with exact cloud vectors.
    """

    def __init__(
        self,
        edge_store: EdgeStore,
        doc_error_map: dict[str, float],
        vector_dim: int,
        bytes_per_float: int = 4,
    ):
        self.edge_store = edge_store
        self.doc_error_map = doc_error_map
        self.vector_dim = vector_dim
        self.bytes_per_vector = vector_dim * bytes_per_float + 64  # include header overhead
        self.hit_counts: dict[str, int] = {}

    def record_hit(self, doc_id: str) -> None:
        """Increment observed query hit count for a document."""
        self.hit_counts[str(doc_id)] = self.hit_counts.get(str(doc_id), 0) + 1

    def record_query_hits(self, doc_ids: list[str]) -> None:
        for did in doc_ids:
            self.record_hit(did)

    def select_repair_candidates(
        self,
        budget_bytes: int,
        policy: str = "priority",  # 'priority', 'frequency', 'error', 'random'
        candidate_ids: list[str] | None = None,
        private_doc_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Select doc IDs to repair under byte budget using specified policy.
        Private docs are strictly excluded from cloud repair.
        """
        private_set = private_doc_ids or set()
        max_vectors = max(1, budget_bytes // self.bytes_per_vector)

        if candidate_ids is None:
            # Consider all known shared docs with errors
            candidate_ids = [d for d in self.doc_error_map.keys() if d not in private_set]
        else:
            candidate_ids = [d for d in candidate_ids if d not in private_set]

        if not candidate_ids:
            return []

        if policy == "random":
            rng = np.random.default_rng(42)
            shuffled = rng.permutation(candidate_ids).tolist()
            return shuffled[:max_vectors]

        elif policy == "frequency":
            # Sort solely by hit count
            sorted_candidates = sorted(
                candidate_ids,
                key=lambda d: self.hit_counts.get(d, 0),
                reverse=True,
            )
            return sorted_candidates[:max_vectors]

        elif policy == "error":
            # Sort solely by cloud precomputed error e_i
            sorted_candidates = sorted(
                candidate_ids,
                key=lambda d: self.doc_error_map.get(d, 0.0),
                reverse=True,
            )
            return sorted_candidates[:max_vectors]

        else:  # 'priority': hit_count * error_i / bytes
            scored = []
            for d in candidate_ids:
                hits = self.hit_counts.get(d, 1)  # floor at 1
                err = self.doc_error_map.get(d, 0.05)
                priority = hits * err
                scored.append((d, priority))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in scored[:max_vectors]]

    def apply_repair(
        self,
        repaired_vectors: dict[str, np.ndarray],
    ) -> int:
        """
        Update local EdgeStore with exact large-model vectors from cloud.
        Sets translated=False.
        """
        if not repaired_vectors:
            return 0

        # Scroll to find current payloads
        doc_payloads = {}
        for ids, _, payloads in self.edge_store.iterate(batch=1024):
            for pid, pld in zip(ids, payloads):
                if str(pid) in repaired_vectors:
                    p = dict(pld)
                    p["translated"] = False
                    p["repaired"] = True
                    doc_payloads[str(pid)] = p

        ids_to_update = list(repaired_vectors.keys())
        vectors_to_update = [repaired_vectors[did] for did in ids_to_update]
        payloads_to_update = [
            doc_payloads.get(did, {"doc_id": did, "translated": False}) for did in ids_to_update
        ]

        self.edge_store.upsert(
            ids=ids_to_update,
            vectors=vectors_to_update,
            payloads=payloads_to_update,
        )
        return len(ids_to_update)
