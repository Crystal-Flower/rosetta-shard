import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from rosetta.certificate import RecallCertificate, extract_features
from rosetta.store import Hit
from rosetta.store.cloud_store import CloudStore
from rosetta.store.edge_store import EdgeStore


class RoutingDecision(str, Enum):
    SERVE_LOCAL = "SERVE_LOCAL"
    ESCALATE = "ESCALATE"
    REFUSE_LOW_CONFIDENCE = "REFUSE_LOW_CONFIDENCE"


@dataclass
class QueryResult:
    query_text: str
    decision: RoutingDecision
    hits: list[Hit]
    certified_overlap: float
    predicted_loss: float
    sim_anchor_max: float
    bytes_transferred: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryRouter:
    """
    Edge query router driven by the Conformal Recall Certificate.
    Enforces the edge-local vs cloud-escalate gating policy.
    """

    def __init__(
        self,
        edge_store: EdgeStore,
        certificate: RecallCertificate,
        cloud_store: CloudStore | None = None,
        tau_serve: float = 0.50,
        alpha: float = 0.10,
        network_available: bool = True,
        k: int = 10,
    ):
        self.edge_store = edge_store
        self.certificate = certificate
        self.cloud_store = cloud_store
        self.tau_serve = tau_serve
        self.alpha = alpha
        self.network_available = network_available
        self.k = k
        self.query_logs: list[QueryResult] = []

    def route_query(
        self,
        query_text: str,
        q_small: np.ndarray,
        q_hat: np.ndarray,
        anchors_centroids: np.ndarray | None = None,
    ) -> QueryResult:
        """
        Execute local query, compute certificate, and route according to confidence & network state.
        """
        start_t = time.perf_counter()
        bytes_transferred = 0

        # 1. Local translated search on EdgeStore
        local_hits = self.edge_store.query(q_hat, k=self.k)

        # 2. Extract features & compute certificate
        features = extract_features(
            q_small=q_small,
            q_hat=q_hat,
            hits=local_hits,
            anchors_centroids=anchors_centroids
            if anchors_centroids is not None
            else self.certificate.anchors_centroids,
            centroid_residuals=self.certificate.centroid_residuals,
            k=self.k,
        )
        cert_overlap, pred_loss = self.certificate.certify(features, alpha=self.alpha)

        # 3. Decision gating
        # OOD gate check: If query is strongly out-of-distribution (sim_anchor_max low), flag risk
        is_ood = features.sim_anchor_max < 0.50
        if is_ood:
            cert_overlap = min(cert_overlap, 0.35)

        if cert_overlap >= self.tau_serve and not is_ood:
            decision = RoutingDecision.SERVE_LOCAL
            final_hits = local_hits
        elif self.network_available and self.cloud_store is not None:
            decision = RoutingDecision.ESCALATE
            # Escalate search to cloud Qdrant
            cloud_hits = self.cloud_store.query(q_hat, k=self.k)
            final_hits = cloud_hits
            # Vector payload bytes (float32 * dim) + header ~ 4 KB
            bytes_transferred = len(q_hat.flatten()) * 4 + 512
        else:
            decision = RoutingDecision.REFUSE_LOW_CONFIDENCE
            final_hits = local_hits

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        res = QueryResult(
            query_text=query_text,
            decision=decision,
            hits=final_hits,
            certified_overlap=cert_overlap,
            predicted_loss=pred_loss,
            sim_anchor_max=features.sim_anchor_max,
            bytes_transferred=bytes_transferred,
            latency_ms=latency_ms,
            metadata={
                "margin_1_10": features.margin_1_10,
                "margin_1_2": features.margin_1_2,
                "is_ood": is_ood,
            },
        )
        self.query_logs.append(res)
        return res
