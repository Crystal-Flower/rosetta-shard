import hashlib

import numpy as np
from fastembed import TextEmbedding

from rosetta.config import EMBED_CACHE_DIR, MODELS, ModelMeta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D numpy array."""
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / max(norm, 1e-12)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


_EMBEDDER_CACHE = {}


def get_embedder(model_name: str) -> TextEmbedding:
    """Get or instantiate cached FastEmbed TextEmbedding instance."""
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = TextEmbedding(model_name=model_name)
    return _EMBEDDER_CACHE[model_name]


def hash_texts(texts: list[str]) -> str:
    """Compute deterministic SHA-256 hash for a list of texts."""
    hasher = hashlib.sha256()
    for t in texts:
        hasher.update(t.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def embed_texts(
    texts: list[str],
    model: str | ModelMeta,
    is_query: bool = False,
    cache_key: str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Embed a list of texts with FastEmbed, applying model-specific prefixes,
    L2-normalization, and persistent disk caching.
    """
    if isinstance(model, str):
        # Resolve from key or raw model name
        if model in MODELS:
            meta = MODELS[model]
        else:
            # Match by short name or full name
            matched = [m for m in MODELS.values() if m.name == model or m.name.endswith(model)]
            if matched:
                meta = matched[0]
            else:
                meta = ModelMeta(name=model, dim=384)
    else:
        meta = model

    # Determine cache file
    text_sig = cache_key or hash_texts(texts)
    prefix_kind = "query" if is_query else "passage"
    safe_name = meta.name.replace("/", "_")
    cache_file = EMBED_CACHE_DIR / f"{safe_name}_{prefix_kind}_{len(texts)}_{text_sig}.npy"

    if cache_file.exists():
        try:
            arr = np.load(cache_file)
            if arr.shape[0] == len(texts) and arr.shape[1] == meta.dim:
                return arr
        except Exception:
            pass  # Corrupted cache, recompute

    prefix = meta.query_prefix if is_query else meta.passage_prefix
    prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

    effective_batch = 16 if meta.dim > 512 else min(batch_size, 64)
    embedder = get_embedder(meta.name)
    # fastembed returns generator of numpy arrays
    gen = embedder.embed(prefixed_texts, batch_size=effective_batch)
    embeddings = list(gen)
    arr = np.array(embeddings, dtype=np.float32)

    # Guarantee L2-normalization
    arr = l2_normalize(arr)

    # Save to disk cache
    try:
        np.save(cache_file, arr)
    except Exception as e:
        print(f"Warning: failed to write embedding cache: {e}")

    return arr
