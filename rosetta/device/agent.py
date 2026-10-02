from pathlib import Path

from rosetta.bundle import AdapterBundle
from rosetta.config import MODELS, ModelMeta
from rosetta.embed import embed_texts
from rosetta.migrate import migrate_shard
from rosetta.router import QueryResult, QueryRouter
from rosetta.space import SpaceGuard
from rosetta.store.cloud_store import CloudStore
from rosetta.store.edge_store import EdgeStore


class DeviceAgent:
    """
    On-device Rosetta agent operating on Qdrant Edge.
    Executes embedded queries, handles Space Guard enforcement, runs atomic migrations,
    and routes queries with Conformal Recall Certificates.
    """

    def __init__(
        self,
        shard_path: str | Path,
        model_name: str = "bge-small-en-v1.5",
        cloud_store: CloudStore | None = None,
        airplane_mode: bool = False,
    ):
        self.shard_path = Path(shard_path).resolve()
        self.model_meta = (
            MODELS[model_name] if model_name in MODELS else ModelMeta(name=model_name, dim=384)
        )
        self.space_id = self.model_meta.space_id
        self.cloud_store = cloud_store
        self.airplane_mode = airplane_mode
        self.active_bundle: AdapterBundle | None = None
        self.router: QueryRouter | None = None

        # Initialize or load EdgeStore
        self.edge_store = EdgeStore(
            path=self.shard_path,
            space_id=self.space_id,
            dim=self.model_meta.dim,
            create_if_missing=True,
        )

    def load_bundle(self, bundle: AdapterBundle) -> None:
        """Load and activate an AdapterBundle on device."""
        if bundle.from_space != self.space_id:
            raise ValueError(
                f"Bundle from_space '{bundle.from_space}' does not match agent resident space '{self.space_id}'"
            )
        self.active_bundle = bundle

        # Instantiate certificate if present in bundle
        if bundle.cert_model and bundle.cert_calib:
            from rosetta.certificate import RecallCertificate

            cert = RecallCertificate.from_dict(
                cert_model=bundle.cert_model,
                cert_calib=bundle.cert_calib,
                anchors_centroids=bundle.anchors_centroids,
            )
            self.router = QueryRouter(
                edge_store=self.edge_store,
                certificate=cert,
                cloud_store=self.cloud_store,
                network_available=not self.airplane_mode,
            )

    def query(
        self,
        query_text: str,
        k: int = 10,
        enforce_space_guard: bool = True,
    ) -> QueryResult:
        """
        Embed query, enforce Space Guard or translate, and route with certificate.
        """
        # Embed on-device using small model
        q_small = embed_texts([query_text], model=self.model_meta, is_query=True)

        # Check Space Guard
        if self.active_bundle is not None:
            # Translated query space
            q_hat = self.active_bundle.apply(q_small)
            current_query_space = self.active_bundle.to_space
        else:
            q_hat = q_small
            current_query_space = self.space_id

        if enforce_space_guard:
            SpaceGuard.validate(current_query_space, self.edge_store.space_id)

        if self.router is not None:
            self.router.network_available = not self.airplane_mode
            return self.router.route_query(
                query_text=query_text,
                q_small=q_small,
                q_hat=q_hat,
            )
        else:
            # Direct search if certificate not yet loaded
            hits = self.edge_store.query(q_hat, k=k)
            from rosetta.router import RoutingDecision

            return QueryResult(
                query_text=query_text,
                decision=RoutingDecision.SERVE_LOCAL,
                hits=hits,
                certified_overlap=1.0,
                predicted_loss=0.0,
                sim_anchor_max=1.0,
                bytes_transferred=0,
                latency_ms=1.0,
            )

    def migrate(self, target_dim: int, batch_size: int = 1024) -> None:
        """Execute atomic shard migration to the new model space."""
        if self.active_bundle is None:
            raise RuntimeError("Cannot migrate without an active AdapterBundle loaded.")

        self.edge_store.close()
        self.edge_store = migrate_shard(
            live_shard_path=self.shard_path,
            bundle=self.active_bundle,
            target_dim=target_dim,
            batch_size=batch_size,
        )
        self.space_id = self.active_bundle.to_space
        if self.router:
            self.router.edge_store = self.edge_store

    def toggle_airplane_mode(self, enabled: bool | None = None) -> bool:
        """Toggle or set airplane mode state."""
        if enabled is not None:
            self.airplane_mode = enabled
        else:
            self.airplane_mode = not self.airplane_mode
        if self.router:
            self.router.network_available = not self.airplane_mode
        return self.airplane_mode

    def close(self) -> None:
        self.edge_store.close()
