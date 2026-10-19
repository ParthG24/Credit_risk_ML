# Credit Risk Prediction

A machine learning project to predict whether a loan applicant will default, using an XGBoost model served through FastAPI with an interactive web UI.

---

## Model Results

Evaluated on a 20% test set (6,482 loans):

| Model | Accuracy | ROC-AUC | F1 Score |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 78.4% | 0.853 | 0.612 |
| Random Forest | 91.3% | 0.929 | 0.791 |
| LightGBM | 92.0% | 0.951 | 0.815 |
| **XGBoost** | **93.8%** | **0.953** | **0.844** |

All experiments and hyperparameter runs were logged locally using MLflow.

---

## How It Works

1. **Preprocessing**: Cleans the loan data, fills missing values, and calculates the loan-to-income ratio.
2. **Prediction**: XGBoost scores the applicant. If the default probability exceeds 56.8%, the loan is marked high risk.
3. **Explanations**: Uses TreeSHAP to show which factors (e.g., high interest rate, income, loan amount) increased or decreased the risk.

---

## Project Structure

```
Credit_risk_ML/
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── credit_risk_pipeline.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── train_and_track.py
│   └── explainability_and_fairness.py
├── app/
│   ├── main.py
│   └── schemas.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── artifacts/
│   ├── model.pkl
│   ├── threshold.pkl
│   ├── preprocessor.pkl
│   └── explainer.pkl
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Clone & install dependencies

```bash
git clone https://github.com/ParthG24/Credit_risk_ML.git
cd Credit_risk_ML

python -m venv venv
venv\Scripts\activate   # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the web app

```bash
python -m uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser.

You can enter applicant details manually or click one of the preset profiles (Prime, High Risk, Borderline) to test the model.

---

## API Endpoints

- `POST /predict`: Returns risk status (`Low Risk` / `High Risk`) and probability.
- `POST /explain`: Returns the prediction along with top factors that affected the score.
- `GET /health`: Basic health check.

---

## License

MIT
