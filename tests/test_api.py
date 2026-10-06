"""API input checks. Nothing here starts a real download or render."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import app

    submitted = []
    monkeypatch.setattr(app.worker, "submit", lambda *a, **k: submitted.append(a))
    c = TestClient(app.app)
    c.submitted = submitted
    return c


def test_rejects_non_links(client):
    r = client.post("/api/jobs", json={"url": "not a link"})
    assert r.status_code == 400
    assert client.submitted == []


def test_queues_a_valid_link_with_clamped_options(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/abc", "num_clips": 99, "min_len": 30, "max_len": 10})
    assert r.status_code == 200
    opts = r.json()["options"]
    assert opts["num_clips"] == 15
    assert opts["max_len"] == 35   # never shorter than min_len + 5
    assert len(client.submitted) == 1


def test_batch_needs_at_least_one_link(client):
    assert client.post("/api/batch", json={"urls": ["  ", ""]}).status_code == 400


def test_story_needs_a_topic(client):
    assert client.post("/api/story", json={"topic": "  "}).status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_only_failed_story_jobs_can_retry(client):
    job = client.post("/api/jobs", json={"url": "https://youtu.be/abc"}).json()
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 400
