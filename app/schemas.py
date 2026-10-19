"""
Pydantic Request and Response Schemas for Credit Risk and XAI Service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Gross annual income in USD/INR")
    person_home_ownership: str = Field(..., description="Home ownership status: RENT, MORTGAGE, OWN, OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, description="Employment duration in years")
    loan_intent: str = Field(..., description="Loan purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Assigned credit grade: A, B, C, D, E, F, G")
    loan_amnt: float = Field(..., gt=0, description="Requested principal loan amount")
    loan_int_rate: float = Field(..., ge=1.0, le=45.0, description="Annual interest rate percentage")
    loan_percent_income: float = Field(..., ge=0.0, le=2.0, description="Loan amount as a fraction of annual income")
    cb_person_default_on_file: str = Field(..., description="Historical credit default recorded: Y or N")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=50, description="Length of bureau credit history in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 75000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 5.0,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "B",
                "loan_amnt": 12000.0,
                "loan_int_rate": 9.25,
                "loan_percent_income": 0.16,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            }
        }
    }


class FactorContribution(BaseModel):
    feature: str
    display_name: str
    impact_value: float
    direction: str  # 'increases_risk' or 'lowers_risk'
    description: str


class PredictionResponse(BaseModel):
    default_probability: float
    default_prediction: int
    risk_verdict: str  # "Low Risk" or "High Risk"
    decision_threshold: float
    confidence_tier: str


class ExplanationResponse(BaseModel):
    default_probability: float
    decision_threshold: float
    risk_verdict: str
    base_value: float
    top_risk_drivers: List[FactorContribution]
    top_protective_factors: List[FactorContribution]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    framework: str
    optimal_threshold: float
