# Credit Risk Scoring & Explainability Engine

A production-grade credit default prediction and explainability service built with XGBoost, Scikit-Learn, TreeSHAP, and FastAPI.

The system trains on 32.5k historical loan records to estimate default risk, applies probability calibration (Platt scaling) to ensure reliability for capital reserve calculations, optimizes decision thresholds for cost-sensitive lending, and computes real-time local feature attributions via TreeSHAP for adverse action notices.

---

## Highlights

- **Predictive Performance**: Evaluates 4 baseline architectures (Logistic Regression, Random Forest, LightGBM, XGBoost); hyperparameter-tuned XGBoost achieves **0.9532 ROC-AUC** and **0.8440 F1-score** on holdout data.
- **Probability Calibration**: Corrects tree-model probability overconfidence using sigmoid scaling (`CalibratedClassifierCV`), lowering Brier score loss from 0.1531 to **0.0542**.
- **Cost-Sensitive Thresholding**: Precision-Recall curve optimization identifies an optimal decision boundary at **0.5683**, balancing loan loss prevention (83.2% recall) with interest margin retention (85.6% precision).
- **Explainable AI (TreeSHAP)**: Sub-20ms local feature attribution via `/explain`, isolating top risk drivers and mitigating factors for regulatory compliance.
- **Algorithmic Fairness Audit**: Fairlearn evaluation across borrower age cohorts (Young `<25` vs Mature `25+`) confirms a **0.0201 Demographic Parity Difference** and passes the four-fifths rule at **97.6%**.
- **Production Serving**: Containerized FastAPI service with Pydantic v2 schemas, automated pytest suite (5/5 passing), and an interactive underwriting dashboard with light/dark modes.

---

## Model Benchmarks

All models were evaluated on an identical stratified 20% holdout test set ($N = 6,482$) with experiment tracking logged via MLflow.

| Model Architecture | ROC-AUC | F1-Score | Precision | Recall | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.8530 | 0.6119 | 0.5312 | 0.7214 | 0.1531 |
| Random Forest (150 trees) | 0.9291 | 0.7910 | 0.7984 | 0.7838 | 0.0819 |
| LightGBM Classifier | 0.9505 | 0.8148 | 0.8041 | 0.8258 | 0.0680 |
| XGBoost (Baseline) | 0.9514 | 0.8189 | 0.8085 | 0.8296 | 0.0627 |
| **Calibrated Tuned XGBoost (Production)** | **0.9532** | **0.8440** | **0.8562** | **0.8321** | **0.0542** |

---

## Project Structure

```
credit-risk-xai-engine/
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI workflow (lint, test, docker build)
├── data/
│   └── credit_risk_dataset.csv # 32.5k loan applications dataset
├── notebooks/
│   └── credit_risk_xai_pipeline.ipynb # Exploratory analysis & modeling pipeline
├── src/
│   ├── data_pipeline.py        # Data cleaning, outlier handling, ColumnTransformer
│   ├── train_and_track.py      # Benchmark training, tuning, calibration, MLflow tracking
│   └── explainability_and_fairness.py # SHAP, LIME, Fairlearn bias audit
├── app/
│   ├── schemas.py              # Pydantic v2 request/response schemas
│   └── main.py                 # FastAPI application & REST endpoints
├── static/
│   ├── index.html              # Underwriting & risk dashboard (Light / Dark mode)
│   ├── style.css               # Clean minimalist interface styles
│   └── script.js               # Dynamic scoring, scale updates, what-if simulator
├── artifacts/
│   ├── model.pkl               # Production calibrated model artifact
│   ├── threshold.pkl           # Optimal F1 decision cutoff (0.5683)
│   ├── preprocessor.pkl        # Fitted Scikit-Learn ColumnTransformer
│   ├── raw_xgb.pkl             # Booster used by TreeSHAP
│   └── explainer.pkl           # Serialized TreeExplainer
├── reports/
│   ├── figures/                # SHAP beeswarm, dependence, disparity charts
│   ├── fairness_audit_report.md # Demographic parity & equalized odds audit
│   └── Responsible_AI.md       # Governance & regulatory checklist
├── tests/
│   └── test_api.py             # Automated pytest suite
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Local container runner
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/ParthG24/credit-risk-xai-engine.git
cd credit-risk-xai-engine

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Start the FastAPI server and Underwriting Dashboard
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` in your browser.

### Docker Deployment

```bash
docker compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Evaluates loan application; returns probability and decision verdict |
| `POST` | `/explain` | Computes local TreeSHAP attribution with top risk drivers |
| `GET` | `/fairness-summary` | Returns Fairlearn disparity metrics across demographic cohorts |
| `GET` | `/health` | Health check and model metadata |

### Sample Inference Request

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 30,
    "person_income": 75000.0,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 5.0,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "B",
    "loan_amnt": 12000.0,
    "loan_int_rate": 8.9,
    "loan_percent_income": 0.16,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
  }'
```

### Sample Response

```json
{
  "default_probability": 0.0832,
  "decision_threshold": 0.5683,
  "risk_verdict": "Low Risk",
  "base_value": 0.2185,
  "top_risk_drivers": [],
  "top_protective_factors": [
    {
      "feature": "person_income",
      "display_name": "Annual Income",
      "impact_value": 0.412,
      "direction": "lowers_risk",
      "description": "Annual Income strengthened creditworthiness"
    },
    {
      "feature": "loan_grade_B",
      "display_name": "Credit Grade: B",
      "impact_value": 0.285,
      "direction": "lowers_risk",
      "description": "Credit Grade: B strengthened creditworthiness"
    }
  ]
}
```

---

## Testing & CI/CD

Run the test suite:
```bash
python -m pytest tests/ -v
```

All pushes and pull requests trigger the GitHub Actions workflow (`.github/workflows/ci.yml`) to run Flake8 code linting, execute Pytest tests, and validate the Docker image build.

---

## License

MIT License.
