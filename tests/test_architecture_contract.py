from fastapi.testclient import TestClient

from backend.main import app


def test_modular_backend_serves_health_and_frontend():
    client = TestClient(app)

    health = client.get("/health")
    frontend = client.get("/app/")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}
    assert frontend.status_code == 200


def test_modular_backend_exposes_task_and_contract_routes():
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])

    assert "/contracts/upload" in paths
    assert "/contracts/{contract_id}/analyze" in paths
    assert "/contracts/{contract_id}/summary" in paths
    assert "/contracts/compare" in paths
    assert "/tasks/{task_id}" in paths
    assert "/chat" in paths


def test_frontend_does_not_call_legacy_api_endpoints():
    app_js = open("frontend/app.js", encoding="utf-8").read()

    legacy_paths = [
        "/analyze_text",
        "/analyze_pdf",
        "/classify_clause",
        "/classify_clauses",
        "/compare_contracts",
    ]

    for path in legacy_paths:
        assert path not in app_js
