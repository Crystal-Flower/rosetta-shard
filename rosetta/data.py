import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi
import numpy as np
import requests

from rosetta.config import DATA_CACHE_DIR, RANDOM_SEED


@dataclass
class Dataset:
    name: str
    corpus: dict[str, str]  # doc_id -> text
    queries: dict[str, str]  # q_id -> text
    qrels: dict[str, dict[str, int]]  # q_id -> {doc_id: relevance}


BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}


def download_and_extract_beir(dataset_name: str) -> Path:
    """Download BEIR dataset zip and extract to cache dir if not present."""
    dest_dir = DATA_CACHE_DIR / dataset_name
    if dest_dir.exists() and (dest_dir / "corpus.jsonl").exists():
        return dest_dir

    if dataset_name not in BEIR_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(BEIR_URLS.keys())}")

    url = BEIR_URLS[dataset_name]
    zip_path = DATA_CACHE_DIR / f"{dataset_name}.zip"

    print(f"Downloading {dataset_name} from {url}...")
    with requests.get(url, stream=True, verify=certifi.where(), timeout=60) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.writelines(r.iter_content(chunk_size=65536))

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_CACHE_DIR)

    if zip_path.exists():
        zip_path.unlink()  # remove zip to save disk

    return dest_dir


def load_dataset(dataset_name: str = "scifact", max_corpus: int | None = 5000) -> Dataset:
    """Load corpus, queries, and qrels for a dataset with deterministic sampling."""
    dataset_dir = download_and_extract_beir(dataset_name)

    # 1. Load corpus
    corpus: dict[str, str] = {}
    with open(dataset_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = str(item["_id"])
            text = (item.get("title", "") + " " + item.get("text", "")).strip()
            corpus[doc_id] = text

    # 2. Load queries
    queries: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[str(item["_id"])] = item["text"].strip()

    # 3. Load qrels (test or default)
    qrels: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        # Fallback to dev or train if test not available
        candidates = list((dataset_dir / "qrels").glob("*.tsv"))
        if candidates:
            qrels_path = candidates[0]

    if qrels_path.exists():
        with open(qrels_path, "r", encoding="utf-8") as f:
            next(f, None)  # skip header if present
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, docid, score = parts[0], parts[1], int(float(parts[2]))
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][docid] = score

    # Filter corpus to max_corpus if requested, keeping all relevant docs for queries
    if max_corpus and len(corpus) > max_corpus:
        relevant_doc_ids = set()
        for doc_scores in qrels.values():
            relevant_doc_ids.update(doc_scores.keys())

        # Always keep relevant documents
        keep_ids = list(relevant_doc_ids & set(corpus.keys()))
        remaining_ids = sorted(list(set(corpus.keys()) - relevant_doc_ids))

        rng = np.random.default_rng(RANDOM_SEED)
        needed = max_corpus - len(keep_ids)
        if needed > 0 and len(remaining_ids) > 0:
            sample_count = min(needed, len(remaining_ids))
            sampled_remaining = rng.choice(remaining_ids, size=sample_count, replace=False)
            keep_ids.extend(sampled_remaining)

        corpus = {doc_id: corpus[doc_id] for doc_id in keep_ids}

    return Dataset(name=dataset_name, corpus=corpus, queries=queries, qrels=qrels)


def split_queries(
    dataset: Dataset,
    n_anchor: int = 200,
    n_dev: int = 100,
    n_cal_train: int = 100,
    n_cal: int = 150,
    n_test: int = 250,
    seed: int = RANDOM_SEED,
) -> dict[str, list[tuple[str, str]]]:
    """
    Split queries into disjoint sets:
    - Q_anchor: trains the adapter
    - Q_dev: chooses hyperparameters (lambda)
    - Q_cal_train: trains the certificate loss predictor
    - Q_cal: calibrates conformal nonconformity quantiles
    - Q_test: held-out evaluation
    """
    all_qids = sorted(list(dataset.queries.keys()))
    rng = np.random.default_rng(seed)
    shuffled_qids = rng.permutation(all_qids).tolist()

    total_needed = n_anchor + n_dev + n_cal_train + n_cal + n_test
    # Scale proportionally if query pool is smaller than total requested
    if len(shuffled_qids) < total_needed:
        scale = len(shuffled_qids) / total_needed
        n_anchor = max(20, int(n_anchor * scale))
        n_dev = max(10, int(n_dev * scale))
        n_cal_train = max(10, int(n_cal_train * scale))
        n_cal = max(20, int(n_cal * scale))
        n_test = len(shuffled_qids) - (n_anchor + n_dev + n_cal_train + n_cal)

    idx = 0
    splits = {}
    for name, count in [
        ("anchor", n_anchor),
        ("dev", n_dev),
        ("cal_train", n_cal_train),
        ("cal", n_cal),
        ("test", n_test),
    ]:
        qids_split = shuffled_qids[idx : idx + count]
        splits[name] = [(qid, dataset.queries[qid]) for qid in qids_split]
        idx += count

    return splits
