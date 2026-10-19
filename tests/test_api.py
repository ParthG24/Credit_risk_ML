"""
Credit Risk Prediction Unit and Integration Test Suite for FastAPI Service.

"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)


def test_health_check():
    """Validates service liveness and metadata response."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "optimal_threshold" in data


def test_predict_prime_borrower():
    """Tests inference endpoint on a prime profile."""
    with TestClient(app) as test_client:
        payload = {
            "person_age": 35,
            "person_income": 95000.0,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 8.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "A",
            "loan_amnt": 10000.0,
            "loan_int_rate": 7.5,
            "loan_percent_income": 0.11,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 10
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "default_probability" in data
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["risk_verdict"] in ["Low Risk", "High Risk"]
        assert data["default_prediction"] in [0, 1]


def test_predict_validation_error():
    """Asserts Pydantic rejects invalid input parameters."""
    with TestClient(app) as test_client:
        invalid_payload = {
            "person_age": 15,  # Invalid: below 18
            "person_income": -500.0,  # Invalid: negative
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 5.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 10000.0,
            "loan_int_rate": 9.5,
            "loan_percent_income": 0.2,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5
        }
        response = test_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422


def test_explain_endpoint():
    """Verifies local TreeSHAP attribution returns ranked risk drivers."""
    with TestClient(app) as test_client:
        payload = {
            "person_age": 29,
            "person_income": 60000.0,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "C",
            "loan_amnt": 15000.0,
            "loan_int_rate": 12.5,
            "loan_percent_income": 0.25,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 6
        }
        response = test_client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "top_risk_drivers" in data
        assert "top_protective_factors" in data
        assert "base_value" in data
        assert len(data["top_risk_drivers"]) <= 3


def test_fairness_summary():
    """Verifies algorithmic fairness metrics endpoint."""
    with TestClient(app) as test_client:
        response = test_client.get("/fairness-summary")
        assert response.status_code == 200
        data = response.json()
        assert "demographic_parity_difference" in data
        assert "equalized_odds_difference" in data
