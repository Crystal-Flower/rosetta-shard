from dataclasses import dataclass
from pathlib import Path

# Global random seed for reproducible benchmarks and splits
RANDOM_SEED: int = 42

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
EMBED_CACHE_DIR = CACHE_DIR / "embeddings"
DATA_CACHE_DIR = CACHE_DIR / "datasets"
BUNDLE_CACHE_DIR = CACHE_DIR / "bundles"
RESULTS_DIR = BASE_DIR / "results"
SHARDS_DIR = BASE_DIR / "data" / "shards"

for d in [EMBED_CACHE_DIR, DATA_CACHE_DIR, BUNDLE_CACHE_DIR, RESULTS_DIR, SHARDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelMeta:
    name: str
    dim: int
    revision: str = "main"
    norm: str = "l2"
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def space_id(self) -> str:
        # Format: <model-name>@<revision>/<dim>/<normalization>
        short_name = self.name.split("/")[-1]
        return f"{short_name}@{self.revision}/{self.dim}/{self.norm}"


# Registered models and their exact prefix rules
MODELS: dict[str, ModelMeta] = {
    "all-MiniLM-L6-v2": ModelMeta(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        query_prefix="",
        passage_prefix="",
    ),
    "bge-small-en-v1.5": ModelMeta(
        name="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
    "bge-large-en-v1.5": ModelMeta(
        name="BAAI/bge-large-en-v1.5",
        dim=1024,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
    "mxbai-embed-large-v1": ModelMeta(
        name="mixedbread-ai/mxbai-embed-large-v1",
        dim=1024,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    ),
}

# Scenarios defined in build spec Section 4
SCENARIOS = {
    "S1": {
        "description": "Silent failure: same dimension (384-d), incompatible coordinate spaces",
        "edge_model": MODELS["all-MiniLM-L6-v2"],
        "cloud_model": MODELS["bge-small-en-v1.5"],
    },
    "S2": {
        "description": "Primary: dimension mismatch (384-d to 1024-d), same model family",
        "edge_model": MODELS["bge-small-en-v1.5"],
        "cloud_model": MODELS["bge-large-en-v1.5"],
    },
    "S3": {
        "description": "Stretch: cross-family dimension mismatch (384-d to 1024-d)",
        "edge_model": MODELS["all-MiniLM-L6-v2"],
        "cloud_model": MODELS["mxbai-embed-large-v1"],
    },
}
