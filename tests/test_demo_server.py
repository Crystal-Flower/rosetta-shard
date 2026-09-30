import pytest
from fastapi.testclient import TestClient

from demo.server import app, state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    # Cleanup demo state after test module
    state.cleanup()


def test_demo_server_lifecycle(client: TestClient):
    # 1. Initial Status
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "device" in data and "cloud" in data
    assert data["device"]["points"] == 8
    assert data["device"]["airplane_mode"] is False
    assert data["device"]["bundle_loaded"] is False

    # 2. Local Query on small model
    resp = client.post("/api/query", json={"query": "battery degradation temperature"})
    assert resp.status_code == 200
    q_data = resp.json()
    assert "hits" in q_data
    assert len(q_data["hits"]) > 0
    assert "certified_overlap" in q_data

    # 3. Toggle Airplane Mode
    resp = client.post("/api/airplane")
    assert resp.status_code == 200
    assert resp.json()["airplane_mode"] is True

    resp = client.post("/api/airplane")
    assert resp.status_code == 200
    assert resp.json()["airplane_mode"] is False

    # 4. Cloud Model Upgrade -> Mismatch
    resp = client.post("/api/upgrade_cloud")
    assert resp.status_code == 200
    assert resp.json()["status"] == "upgraded"

    # 5. Query blocked by Space Guard
    resp = client.post("/api/query", json={"query": "neural network quantization"})
    assert resp.status_code == 400
    err_data = resp.json()
    assert err_data["error"] == "SpaceMismatchError"
    assert err_data["blocked"] is True

    # 6. Push Adapter Bundle
    resp = client.post("/api/push_adapter")
    assert resp.status_code == 200
    assert resp.json()["status"] == "adapter_loaded"

    # 7. Migrate Edge Shard
    resp = client.post("/api/migrate")
    assert resp.status_code == 200
    mig_data = resp.json()
    assert mig_data["points_migrated"] == 8
    assert "speedup_factor" in mig_data

    # 8. Query post-migration (now matches upgraded cloud space)
    resp = client.post("/api/query", json={"query": "neural network quantization"})
    assert resp.status_code == 200
    post_data = resp.json()
    assert len(post_data["hits"]) > 0

    # 9. Reset Demo
    resp = client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset_complete"
