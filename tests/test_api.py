"""HTTP-level smoke tests for primary API routes."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_APP = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5_000_000,
    "loan_amount": 12_000_000,
    "loan_term": 120,
    "cibil_score": 720,
    "residential_assets_value": 2_000_000,
    "commercial_assets_value": 1_000_000,
    "luxury_assets_value": 500_000,
    "bank_asset_value": 800_000,
}


def test_health():
    """Root health endpoint responds OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_risk_smoke():
    """predict-risk returns structured payload when models exist."""
    response = client.post("/api/v1/predict-risk", json=SAMPLE_APP)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "risk_score" in body["data"]
    assert body["data"]["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_decision_endpoint():
    """Decision endpoint produces APPROVE/REVIEW/REJECT."""
    response = client.post(
        "/api/v1/decision",
        json={"application": SAMPLE_APP},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] in {"APPROVE", "REVIEW", "REJECT"}


def test_trust_score_endpoint():
    """trust-score returns 0–100 score and category."""
    response = client.post("/api/v1/trust-score", json=SAMPLE_APP)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert "trust_score" in data
    assert 0 <= data["trust_score"] <= 100
    assert data["trust_category"] in {"STRONG", "MODERATE", "WEAK"}


def test_explain_endpoint():
    """explain returns ranked features and narrative."""
    response = client.post("/api/v1/explain", json=SAMPLE_APP)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert "top_features" in data
    assert "summary" in data
    assert isinstance(data["top_features"], list)


def test_simulate_endpoint():
    """simulate returns baseline and scenario rows."""
    response = client.post("/api/v1/simulate", json=SAMPLE_APP)
    assert response.status_code == 200
    payload = response.json()
    assert "baseline_risk_score" in payload
    assert "baseline_decision" in payload
    assert len(payload["scenarios"]) >= 1
    for row in payload["scenarios"]:
        assert "risk_score" in row
        assert row["decision"] in {"APPROVE", "REVIEW", "REJECT"}


def test_recommend_endpoint():
    """recommend returns a non-empty suggestions list."""
    response = client.post("/api/v1/recommend", json=SAMPLE_APP)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    suggestions = body["data"]["suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1
    assert "title" in suggestions[0]


def test_compare_scenarios_endpoint():
    """compare-scenarios ranks labeled alternatives."""
    payload = {
        "base_application": SAMPLE_APP,
        "scenarios": [
            {"label": "Lower loan", "income_annum": 5_000_000, "loan_amount": 8_000_000, "cibil_score": 720},
            {"label": "Higher CIBIL", "income_annum": 5_000_000, "loan_amount": 12_000_000, "cibil_score": 780},
        ],
    }
    response = client.post("/api/v1/compare-scenarios", json=payload)
    assert response.status_code == 200
    out = response.json()
    assert "results" in out
    assert "best_option_label" in out
    assert len(out["results"]) == 2
    assert out["results"][0]["rank"] == 1
