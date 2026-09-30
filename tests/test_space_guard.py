import numpy as np
import pytest

from rosetta.bundle import AdapterBundle
from rosetta.space import AdapterRegistry, QueryVector, SpaceGuard, SpaceMismatchError


def test_space_guard_validation():
    # Same space passes
    SpaceGuard.validate("modelA@main/384/l2", "modelA@main/384/l2")

    # Mismatched space raises
    with pytest.raises(SpaceMismatchError) as exc_info:
        SpaceGuard.validate("modelA@main/384/l2", "modelB@main/384/l2")

    assert "modelA@main/384/l2" in str(exc_info.value)
    assert "modelB@main/384/l2" in str(exc_info.value)
    assert exc_info.value.query_space == "modelA@main/384/l2"
    assert exc_info.value.store_space == "modelB@main/384/l2"


def test_query_vector_normalization():
    raw_vec = np.array([3.0, 4.0], dtype=np.float32)
    qv = QueryVector(vector=raw_vec, space_id="test_space")
    assert qv.space_id == "test_space"
    np.testing.assert_allclose(np.linalg.norm(qv.vector), 1.0, atol=1e-5)
    np.testing.assert_allclose(qv.vector, np.array([0.6, 0.8], dtype=np.float32), atol=1e-5)


def test_adapter_registry_single_and_multi_hop():
    reg = AdapterRegistry()

    # Create dummy bundles for A -> B and B -> C
    W_ab = np.eye(4, dtype=np.float32)
    bundle_ab = AdapterBundle(
        from_space="space_a",
        to_space="space_b",
        W=W_ab,
    )

    W_bc = 2.0 * np.eye(4, dtype=np.float32)
    bundle_bc = AdapterBundle(
        from_space="space_b",
        to_space="space_c",
        W=W_bc,
    )

    reg.register(bundle_ab)
    reg.register(bundle_bc)

    # 1. Identity when query already in target space
    q_a = QueryVector(vector=np.array([1.0, 0.0, 0.0, 0.0]), space_id="space_a")
    same_q = reg.translate(q_a, "space_a")
    assert same_q.space_id == "space_a"
    np.testing.assert_allclose(same_q.vector, q_a.vector)

    # 2. Single-hop: A -> B
    q_b = reg.translate(q_a, "space_b")
    assert q_b.space_id == "space_b"
    np.testing.assert_allclose(np.linalg.norm(q_b.vector), 1.0, atol=1e-5)

    # 3. Multi-hop: A -> C
    q_c = reg.translate(q_a, "space_c")
    assert q_c.space_id == "space_c"
    np.testing.assert_allclose(np.linalg.norm(q_c.vector), 1.0, atol=1e-5)

    # 4. Unknown destination raises ValueError
    with pytest.raises(ValueError):
        reg.translate(q_a, "space_unknown")
