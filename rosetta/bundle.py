import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosetta.adapters.base import l2_normalize


class AdapterBundle:
    """
    Portable, verifiable Adapter Bundle package shipped cloud-to-device.
    Contains:
    - manifest.json: provenance, space IDs, SHA-256 integrity hashes
    - W.npy: translation matrix
    - mu_x.npy, mu_y.npy: optional centering means
    - anchors_centroids.npy: anchor centroids in small space for OOD detection
    - doc_error_map.npy: per-document cloud-computed translation residuals
    - cert_model.json: loss predictor coefficients
    - cert_calib.json: conformal nonconformity quantiles
    """

    def __init__(
        self,
        from_space: str,
        to_space: str,
        W: np.ndarray,
        mu_x: np.ndarray | None = None,
        mu_y: np.ndarray | None = None,
        anchors_centroids: np.ndarray | None = None,
        doc_error_map: dict[str, float] | None = None,
        cert_model: dict[str, Any] | None = None,
        cert_calib: dict[str, Any] | None = None,
        git_sha: str = "main",
    ):
        self.from_space = from_space
        self.to_space = to_space
        self.W = np.asarray(W, dtype=np.float32)
        self.mu_x = np.asarray(mu_x, dtype=np.float32) if mu_x is not None else None
        self.mu_y = np.asarray(mu_y, dtype=np.float32) if mu_y is not None else None
        self.anchors_centroids = (
            np.asarray(anchors_centroids, dtype=np.float32)
            if anchors_centroids is not None
            else None
        )
        self.doc_error_map = doc_error_map or {}
        self.cert_model = cert_model or {}
        self.cert_calib = cert_calib or {}
        self.git_sha = git_sha
        self.created_at = datetime.datetime.utcnow().isoformat() + "Z"

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Translate source vectors X into target space with unit L2 normalization."""
        X = l2_normalize(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xc = X - self.mu_x if self.mu_x is not None else X
        proj = Xc @ self.W
        if self.mu_y is not None:
            proj = proj + self.mu_y
        return l2_normalize(proj)

    def save(self, bundle_dir: str | Path) -> Path:
        """Save bundle files and manifest with SHA-256 checksums."""
        out_dir = Path(bundle_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        files_to_hash = {}

        # 1. W.npy
        w_path = out_dir / "W.npy"
        np.save(w_path, self.W)
        files_to_hash["W.npy"] = self._file_sha256(w_path)

        # 2. mu_x.npy, mu_y.npy
        if self.mu_x is not None:
            mx_path = out_dir / "mu_x.npy"
            np.save(mx_path, self.mu_x)
            files_to_hash["mu_x.npy"] = self._file_sha256(mx_path)

        if self.mu_y is not None:
            my_path = out_dir / "mu_y.npy"
            np.save(my_path, self.mu_y)
            files_to_hash["mu_y.npy"] = self._file_sha256(my_path)

        # 3. anchors_centroids.npy
        if self.anchors_centroids is not None:
            anc_path = out_dir / "anchors_centroids.npy"
            np.save(anc_path, self.anchors_centroids)
            files_to_hash["anchors_centroids.npy"] = self._file_sha256(anc_path)

        # 4. doc_error_map.json
        err_path = out_dir / "doc_error_map.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_error_map, f)
        files_to_hash["doc_error_map.json"] = self._file_sha256(err_path)

        # 5. cert_model.json
        cm_path = out_dir / "cert_model.json"
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(self.cert_model, f, indent=2)
        files_to_hash["cert_model.json"] = self._file_sha256(cm_path)

        # 6. cert_calib.json
        cc_path = out_dir / "cert_calib.json"
        with open(cc_path, "w", encoding="utf-8") as f:
            json.dump(self.cert_calib, f, indent=2)
        files_to_hash["cert_calib.json"] = self._file_sha256(cc_path)

        # 7. manifest.json
        manifest = {
            "from_space": self.from_space,
            "to_space": self.to_space,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "checksums": files_to_hash,
            "bundle_size_kb": round(self.size_kb(out_dir), 2),
        }
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_dir

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "AdapterBundle":
        """Load and verify AdapterBundle from directory."""
        b_dir = Path(bundle_dir).resolve()
        manifest_path = b_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {bundle_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Verify checksums
        for fname, expected_hash in manifest.get("checksums", {}).items():
            fpath = b_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Bundle file missing: {fname}")
            actual_hash = cls._file_sha256(fpath)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                )

        W = np.load(b_dir / "W.npy")
        mu_x = np.load(b_dir / "mu_x.npy") if (b_dir / "mu_x.npy").exists() else None
        mu_y = np.load(b_dir / "mu_y.npy") if (b_dir / "mu_y.npy").exists() else None
        anchors_centroids = (
            np.load(b_dir / "anchors_centroids.npy")
            if (b_dir / "anchors_centroids.npy").exists()
            else None
        )

        doc_error_map = {}
        if (b_dir / "doc_error_map.json").exists():
            with open(b_dir / "doc_error_map.json", "r", encoding="utf-8") as f:
                doc_error_map = json.load(f)

        cert_model = {}
        if (b_dir / "cert_model.json").exists():
            with open(b_dir / "cert_model.json", "r", encoding="utf-8") as f:
                cert_model = json.load(f)

        cert_calib = {}
        if (b_dir / "cert_calib.json").exists():
            with open(b_dir / "cert_calib.json", "r", encoding="utf-8") as f:
                cert_calib = json.load(f)

        return cls(
            from_space=manifest["from_space"],
            to_space=manifest["to_space"],
            W=W,
            mu_x=mu_x,
            mu_y=mu_y,
            anchors_centroids=anchors_centroids,
            doc_error_map=doc_error_map,
            cert_model=cert_model,
            cert_calib=cert_calib,
            git_sha=manifest.get("git_sha", "main"),
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def size_kb(bundle_dir: Path) -> float:
        total = sum(f.stat().st_size for f in bundle_dir.glob("*") if f.is_file())
        return total / 1024.0
