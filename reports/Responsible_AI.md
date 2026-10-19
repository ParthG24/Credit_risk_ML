# Responsible AI & Algorithmic Governance Framework
## Credit Risk Intelligence Platform

### 1. Model Identity & Scope
- **System Purpose**: Automated consumer credit underwriting risk assessment and default probability estimation.
- **Model Family**: Gradient Boosted Decision Trees (XGBoost) with Sigmoid Probability Calibration (`CalibratedClassifierCV`).
- **Target Variable**: Binary loan default status within loan tenure (`0 = Non-Default / Paid`, `1 = Charged-Off / Default`).
- **Operating Threshold**: $T^* = 0.5683$ (mathematically optimized to maximize $F_1$ harmonic mean on calibrated validation data).

---

### 2. Fair Lending & Non-Discrimination Audit
In compliance with the **US Equal Credit Opportunity Act (ECOA)** and the **EU AI Act (High-Risk AI Systems - Annex III)**:

1. **Excluded Protected Classes**:
   - Explicit protected attributes (Race, Religion, Sex, Marital Status, National Origin) are **strictly excluded** from training features and model inference.
2. **Sensitive Cohort Auditing (Age Proxy)**:
   - Evaluated across borrower age cohorts: Young (<25) vs Mature (25+).
   - **Demographic Parity Difference**: `0.0201` (Acceptable tolerance: `< 0.10`).
   - **Equalized Odds Difference**: `0.0579` (Acceptable tolerance: `< 0.10`).
   - **Four-Fifths Rule (80% Rule)**: Selection Rate Ratio is **97.6%**, well above the 80% legal threshold.

---

### 3. Model Explainability & Transparency (XAI)
To satisfy **Fair Credit Reporting Act (FCRA) Section 615(a)** requirements for **Adverse Action Notices**:
1. **Global Transparency (SHAP TreeExplainer)**:
   - Feature attributions calculated using cooperative game-theoretic Shapley values.
   - Identified primary systemic default risk drivers:
     - `loan_percent_income` (Loan-to-income ratio)
     - `loan_int_rate` (Subprime risk pricing)
     - `cb_person_default_on_file_Y` (Prior bureau derogatory records)
2. **Local Adverse Action Attribution**:
   - For every denied or high-risk application, the API endpoint `/explain` isolates the top 3 contributing risk factors.
   - Plain-language disclosure provided directly to human underwriting officers and applicants.

---

### 4. Data Privacy & Leakage Controls
1. **Personally Identifiable Information (PII)**:
   - No direct identifiers (SSN, Full Name, IP Address, Physical Address) are ingested into the feature store.
2. **Temporal & Target Leakage Safeguards**:
   - Imputation statistics (median income, employment length) and One-Hot Encoding mappings are fit strictly on training splits within an isolated `Pipeline` to prevent test contamination.
   - Post-loan performance metrics (e.g., actual recovery, collections) are excluded from the training feature set.

---

### 5. Continuous Monitoring & Production Governance
- **Data & Concept Drift**: Monitored via Population Stability Index (PSI) with thresholds:
  - $\text{PSI} < 0.10$: Stable portfolio distribution.
  - $0.10 \le \text{PSI} < 0.25$: Warning, scheduled retraining review.
  - $\text{PSI} \ge 0.25$: Trigger automated alert and model fallback.
- **Human-in-the-Loop Review**:
  - Applications scoring in the borderline band ($0.45 \le P(\text{default}) < 0.60$) are flagged for senior human underwriter manual review.
