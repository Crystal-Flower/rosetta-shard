import math


def overlap_at_k(a_ids: list[str | int], b_ids: list[str | int], k: int = 10) -> float:
    """
    Measure retrieval set fidelity between two candidate result lists (e.g. translated vs native ceiling).
    Range: [0.0, 1.0]
    """
    if k <= 0:
        return 0.0
    set_a = set(a_ids[:k])
    set_b = set(b_ids[:k])
    return len(set_a & set_b) / float(k)


def loss_at_k(a_ids: list[str | int], b_ids: list[str | int], k: int = 10) -> float:
    """Translation loss on top-k retrieval: 1 - overlap@k."""
    return 1.0 - overlap_at_k(a_ids, b_ids, k=k)


def recall_at_k(
    retrieved_ids: list[str | int],
    relevant_ids: set[str | int] | dict[str | int, int],
    k: int = 10,
) -> float:
    """Compute Recall@k against ground-truth relevant doc IDs."""
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids.keys()) if isinstance(relevant_ids, dict) else set(relevant_ids)
    if not rel_set:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & rel_set)
    return hits / float(len(rel_set))


def ndcg_at_k(
    retrieved_ids: list[str | int],
    relevance_scores: dict[str | int, int],
    k: int = 10,
) -> float:
    """Compute normalized Discounted Cumulative Gain (nDCG@k)."""
    if not relevance_scores or k <= 0:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)

    # Ideal DCG@k
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg
