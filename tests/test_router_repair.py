import os
import shutil
import tempfile

import numpy as np

from rosetta.certificate import RecallCertificate
from rosetta.repair import HotSetRepair
from rosetta.router import QueryRouter, RoutingDecision
from rosetta.store import Hit
from rosetta.store.edge_store import EdgeStore


class MockCloudStore:
    def __init__(self, space_id: str):
        self.space_id = space_id

    def query(self, vector, k=10, space_id=None):
        return [Hit(id=f"cloud_doc_{i}", score=0.99 - 0.01 * i, payload={}) for i in range(k)]


def test_query_router_decisions():
    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "router_shard")
        store = EdgeStore(path=shard_path, space_id="test@main/4/l2", dim=4)
        store.upsert(
            ids=[f"doc_{i}" for i in range(5)],
            vectors=np.eye(5, 4, dtype=np.float32),
            payloads=[{"doc_id": f"doc_{i}"} for i in range(5)],
        )

        cloud_store = MockCloudStore("test@main/4/l2")

        # Create dummy certificate
        cert = RecallCertificate(
            weights=np.zeros(6, dtype=np.float32),
            intercept=0.1,  # predicted loss = 0.1 -> overlap ~ 0.9 - q_hat
            quantiles={"0.10": 0.05},
            feature_means=np.zeros(6, dtype=np.float32),
            feature_stds=np.ones(6, dtype=np.float32),
        )

        router = QueryRouter(
            edge_store=store,
            certificate=cert,
            cloud_store=cloud_store,
            tau_serve=0.70,
            alpha=0.10,
            network_available=True,
        )

        q_dummy = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 1. High confidence -> SERVE_LOCAL
        res_local = router.route_query("high conf query", q_small=q_dummy, q_hat=q_dummy)
        assert res_local.decision == RoutingDecision.SERVE_LOCAL
        assert res_local.bytes_transferred == 0

        # 2. Low confidence with network -> ESCALATE
        router.tau_serve = 0.99  # higher than certified overlap
        res_esc = router.route_query("low conf query", q_small=q_dummy, q_hat=q_dummy)
        assert res_esc.decision == RoutingDecision.ESCALATE
        assert res_esc.bytes_transferred > 0
        assert "cloud_doc" in res_esc.hits[0].id

        # 3. Low confidence without network -> REFUSE_LOW_CONFIDENCE
        router.network_available = False
        res_refuse = router.route_query("offline query", q_small=q_dummy, q_hat=q_dummy)
        assert res_refuse.decision == RoutingDecision.REFUSE_LOW_CONFIDENCE

        store.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hot_set_repair():
    tmp_dir = tempfile.mkdtemp()
    try:
        shard_path = os.path.join(tmp_dir, "repair_shard")
        store = EdgeStore(path=shard_path, space_id="test@main/4/l2", dim=4)
        store.upsert(
            ids=["doc_1", "doc_2", "doc_3", "doc_private"],
            vectors=np.eye(4, dtype=np.float32),
            payloads=[
                {"doc_id": "doc_1", "origin": "shared", "translated": True},
                {"doc_id": "doc_2", "origin": "shared", "translated": True},
                {"doc_id": "doc_3", "origin": "shared", "translated": True},
                {"doc_id": "doc_private", "origin": "private", "translated": True},
            ],
        )

        error_map = {
            "doc_1": 0.05,
            "doc_2": 0.40,
            "doc_3": 0.20,
        }

        repair = HotSetRepair(
            edge_store=store,
            doc_error_map=error_map,
            vector_dim=4,
            bytes_per_float=4,
        )

        # Simulate user queries hitting doc_3 frequently
        repair.record_query_hits(["doc_3", "doc_3", "doc_3", "doc_1"])

        # Check budget sizing
        budget_for_1 = repair.bytes_per_vector + 10
        candidates = repair.select_repair_candidates(
            budget_bytes=budget_for_1,
            policy="priority",
            private_doc_ids={"doc_private"},
        )
        assert len(candidates) == 1
        # doc_3 has 3 hits * 0.20 = 0.60 priority vs doc_2: 1 * 0.40 = 0.40
        assert candidates[0] == "doc_3"

        # Check private doc strictly excluded even if provided
        cand_with_priv = repair.select_repair_candidates(
            budget_bytes=10000,
            policy="priority",
            candidate_ids=["doc_1", "doc_2", "doc_private"],
            private_doc_ids={"doc_private"},
        )
        assert "doc_private" not in cand_with_priv

        # Apply exact cloud vector
        new_vec_3 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        n_rep = repair.apply_repair({"doc_3": new_vec_3})
        assert n_rep == 1

        # Verify payload updated in store
        for b_ids, b_vecs, b_payloads in store.iterate():
            for pid, vec, pld in zip(b_ids, b_vecs, b_payloads):
                if pid == "doc_3":
                    assert pld["translated"] is False
                    assert pld["repaired"] is True
                    np.testing.assert_allclose(
                        vec, new_vec_3 / np.linalg.norm(new_vec_3), atol=1e-5
                    )

        store.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
