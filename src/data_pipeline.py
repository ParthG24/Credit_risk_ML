"""
Data Pipeline and Feature Engineering Module for Credit Risk Assessment.
Handles raw dataset ingestion, outlier filtering, feature engineering,
and preprocessor pipeline construction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

SENSITIVE_FEATURE = 'age_group'


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Ingests and cleans credit loan records.
    Filters out invalid logical records while properly preserving
    missing employment values for downstream median imputation.
    """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates().reset_index(drop=True)

    # Valid physiological and regulatory age boundaries
    df = df[(df['person_age'] >= 18) & (df['person_age'] <= 85)]

    # Preserve rows with null employment length, but drop impossible work experience
    valid_emp_mask = df['person_emp_length'].isna() | (
        (df['person_emp_length'] <= df['person_age'] - 14) & (df['person_emp_length'] <= 50)
    )
    df = df[valid_emp_mask]

    # Valid financial records: positive income and positive loan requests
    df = df[(df['person_income'] > 0) & (df['loan_amnt'] > 0)]

    # Add sensitive demographic attribute for Fair Lending audits
    # Young borrowers (< 25) vs Established borrowers (>= 25)
    df['age_group'] = np.where(df['person_age'] < 25, 'Young (<25)', 'Mature (25+)')

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for numerical imputation and categorical encoding.
    Tree-based models (XGBoost, LightGBM, Random Forest) operate on unscaled features,
    while linear models can share this transformer with optional scaling.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def get_train_test_data(filepath: str, test_size: float = 0.20, random_state: int = 42):
    """
    Returns train and test splits with stratified sampling based on default status.
    Also returns the sensitive demographic series for fairness auditing.
    """
    df = load_and_clean_data(filepath)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df['loan_status'].astype(int)
    sensitive_attr = df[SENSITIVE_FEATURE]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_attr,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, sens_train, sens_test
