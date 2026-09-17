"""
Mock, no-GPU integration test.

Verifies the full vertical slice through the real HTTP API:

    create project -> upload video -> processing -> COMPLETED
        -> model available
        -> semantic objects returned
        -> analytics returned
        -> results aggregate returned

Everything runs with USE_MOCK_*=true so it passes on any machine with no
GPU, no COLMAP, and no model weights.
"""

from __future__ import annotations

import io
import os
import time

import pytest

# Force full mock mode BEFORE importing the app / settings (cached).
os.environ.setdefault("USE_MOCK_VISION", "true")
os.environ.setdefault("USE_MOCK_RECONSTRUCTION", "true")
os.environ.setdefault("USE_MOCK_AI", "true")
os.environ.setdefault("USE_MOCK_ANALYTICS", "true")

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _wait_for_completion(client: TestClient, project_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/projects/{project_id}/status")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in {"COMPLETED", "FAILED"}:
            return last
        time.sleep(0.2)
    raise AssertionError(f"Timed out. Last status: {last}")


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mocks"]["use_mock_reconstruction"] is True


def test_full_pipeline_mock(client: TestClient):
    # 1. Create
    resp = client.post("/api/projects", json={"name": "integration-test"})
    assert resp.status_code == 200
    project_id = resp.json()["project_id"]
    assert project_id

    # 2. Upload (triggers background processing). TestClient runs the
    #    background task synchronously after the response is returned.
    fake_video = io.BytesIO(b"not a real video, mock mode ignores it")
    resp = client.post(
        f"/api/projects/{project_id}/upload",
        files={"video": ("test.mp4", fake_video, "video/mp4")},
    )
    assert resp.status_code == 200

    # 3. Wait for COMPLETED
    status = _wait_for_completion(client, project_id)
    assert status["status"] == "COMPLETED", f"pipeline failed: {status.get('error')}"
    assert status.get("error") is None

    # 4. Model available + downloadable
    info = client.get(f"/api/projects/{project_id}/model").json()
    assert info["available"] is True
    model_resp = client.get(f"/api/projects/{project_id}/model/file")
    assert model_resp.status_code == 200
    assert model_resp.content[:4] == b"glTF"  # GLB magic bytes

    # 5. Semantic objects
    objects = client.get(f"/api/projects/{project_id}/semantic").json()
    assert isinstance(objects, list)
    assert len(objects) == 39  # demo AI dataset
    sample = objects[0]
    assert "class" in sample  # alias preserved
    assert sample["category"] in {"Vehicle", "Building", "Person", "Vegetation"}

    # 6. Analytics
    analytics = client.get(f"/api/projects/{project_id}/analytics").json()
    assert analytics["objects"] == 39
    assert analytics["vehicles"] == 12
    assert analytics["buildings"] == 4
    assert analytics["people"] == 7
    assert analytics["vegetation"] == 16
    assert 0 <= analytics["average_detection_confidence"] <= 1

    # 7. Aggregated results
    results = client.get(f"/api/projects/{project_id}/results").json()
    assert results["status"] == "COMPLETED"
    assert results["model_url"]
    assert len(results["objects"]) == 39
    assert results["analytics"]["objects"] == 39

    # 8. Report endpoint
    report = client.get(f"/api/projects/{project_id}/report").json()
    assert report["project_id"] == project_id
    assert report["status"] == "COMPLETED"


def test_unknown_project_returns_404(client: TestClient):
    resp = client.get("/api/projects/does_not_exist/status")
    assert resp.status_code == 404
