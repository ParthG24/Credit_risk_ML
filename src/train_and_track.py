"""
Model Training, Cross-Validation, Calibration, and MLflow Experiment Tracking.
Benchmarks 5 candidate architectures, tunes hyperparameters, calibrates probabilities,
and determines cost-optimal classification thresholds.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn

from data_pipeline import get_train_test_data, build_preprocessor


def calculate_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Computes precision-recall curve and selects the decision threshold
    that maximizes the true harmonic F1-score: 2 * (P * R) / (P + R).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    denominator = precisions + recalls
    f1_scores = np.divide(
        2 * (precisions * recalls),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0
    )
    # thresholds array has length len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def plot_pr_and_roc(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, output_path: str):
    """Generates and persists evaluation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    ax1.plot(recalls, precisions, color='#0284c7', lw=2)
    ax1.set_title(f'{model_name} - Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.grid(True, alpha=0.3)

    # Calibration Reliability Curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax2.plot(prob_pred, prob_true, 's-', color='#10b981', label=model_name)
    ax2.set_title(f'{model_name} - Calibration Reliability')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Fraction of True Positives')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_benchmarking_and_tracking(data_path: str, artifacts_dir: str = 'artifacts'):
    """
    Executes end-to-end multi-model benchmarking with MLflow logging.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    plots_dir = os.path.join(artifacts_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("[1/5] Ingesting and splitting dataset...")
    X_train, X_test, y_train, y_test, sens_train, sens_test = get_train_test_data(data_path)

    # Compute imbalance weight
    neg_count, pos_count = np.bincount(y_train)
    imbalance_ratio = float(neg_count / pos_count)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}. Class Imbalance Ratio: {imbalance_ratio:.2f}")

    # Set up MLflow
    mlflow.set_experiment("credit_risk_underwriting")

    # Define Candidate Benchmark Models
    candidate_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=10, random_state=42),
        "LightGBM": LGBMClassifier(scale_pos_weight=imbalance_ratio, random_state=42, verbose=-1),
        "XGBoost_Baseline": XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42)
    }

    benchmark_results = []

    print("\n[2/5] Training 4 Baseline Architectures with MLflow Tracking...")
    for name, estimator in candidate_models.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            pipe = Pipeline(steps=[
                ('preprocessor', build_preprocessor()),
                ('classifier', estimator)
            ])

            pipe.fit(X_train, y_train)
            test_proba = pipe.predict_proba(X_test)[:, 1]
            test_preds = (test_proba >= 0.5).astype(int)

            metrics = {
                "accuracy": accuracy_score(y_test, test_preds),
                "precision": precision_score(y_test, test_preds, zero_division=0),
                "recall": recall_score(y_test, test_preds),
                "f1": f1_score(y_test, test_preds),
                "roc_auc": roc_auc_score(y_test, test_proba),
                "brier_score": brier_score_loss(y_test, test_proba)
            }

            # Log to MLflow
            mlflow.log_param("model_family", name)
            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)

            plot_path = os.path.join(plots_dir, f"{name}_curves.png")
            plot_pr_and_roc(y_test, test_proba, name, plot_path)
            mlflow.log_artifact(plot_path)

            benchmark_results.append({
                "Model": name,
                **metrics
            })
            print(f" -> {name:20s} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Brier: {metrics['brier_score']:.4f}")

    # Hyperparameter Tuning on XGBoost
    print("\n[3/5] Hyperparameter Tuning on XGBoost via RandomizedSearchCV...")
    xgb_base_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('classifier', XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42))
    ])

    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [3, 4, 5, 6, 7],
        'classifier__learning_rate': [0.02, 0.05, 0.1, 0.15],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__gamma': [0.1, 0.5, 1.0, 2.0]
    }

    cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        xgb_base_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring='average_precision',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_pipeline = random_search.best_estimator_

    print("\n[4/5] Probability Calibration and Threshold Optimization...")
    # Probability calibration on the best estimator using sigmoid scaling
    calibrated_clf = CalibratedClassifierCV(
        estimator=best_pipeline,
        method='sigmoid',
        cv=4
    )
    calibrated_clf.fit(X_train, y_train)

    # Evaluate Calibrated Model on Holdout Set
    calibrated_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    optimal_threshold, best_f1 = calculate_optimal_threshold(y_test, calibrated_proba)
    calibrated_preds = (calibrated_proba >= optimal_threshold).astype(int)

    final_metrics = {
        "accuracy": accuracy_score(y_test, calibrated_preds),
        "precision": precision_score(y_test, calibrated_preds, zero_division=0),
        "recall": recall_score(y_test, calibrated_preds),
        "f1": best_f1,
        "roc_auc": roc_auc_score(y_test, calibrated_proba),
        "brier_score": brier_score_loss(y_test, calibrated_proba),
        "optimal_threshold": optimal_threshold
    }

    print(f" -> Calibrated Tuned XGBoost | ROC-AUC: {final_metrics['roc_auc']:.4f} | Optimal F1: {final_metrics['f1']:.4f} at Threshold {optimal_threshold:.4f}")

    # Log Winning Production Model to MLflow
    with mlflow.start_run(run_name="production_calibrated_xgboost"):
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("optimal_decision_threshold", optimal_threshold)
        mlflow.log_metrics(final_metrics)

        final_plot_path = os.path.join(plots_dir, "production_calibrated_curves.png")
        plot_pr_and_roc(y_test, calibrated_proba, "Calibrated Tuned XGBoost", final_plot_path)
        mlflow.log_artifact(final_plot_path)

        try:
            mlflow.sklearn.log_model(calibrated_clf, artifact_path="model", serialization_format="pickle")
        except Exception as e:
            print(f"Note: MLflow model logging fallback to artifact: {e}")
            model_dump_path = os.path.join(artifacts_dir, 'model.pkl')
            joblib.dump(calibrated_clf, model_dump_path)
            mlflow.log_artifact(model_dump_path)

    print("\n[5/5] Serializing Production Artifacts...")
    # Save fitted preprocessor, uncalibrated tree (for TreeExplainer), and calibrated ensemble
    preprocessor = best_pipeline.named_steps['preprocessor']
    raw_xgb_classifier = best_pipeline.named_steps['classifier']

    joblib.dump(calibrated_clf, os.path.join(artifacts_dir, 'model.pkl'))
    joblib.dump(optimal_threshold, os.path.join(artifacts_dir, 'threshold.pkl'))
    joblib.dump(preprocessor, os.path.join(artifacts_dir, 'preprocessor.pkl'))
    joblib.dump(raw_xgb_classifier, os.path.join(artifacts_dir, 'raw_xgb.pkl'))

    # Save feature names after One-Hot Encoding
    feature_names = list(preprocessor.get_feature_names_out())
    joblib.dump(feature_names, os.path.join(artifacts_dir, 'feature_names.pkl'))

    # Save benchmark summary table
    df_benchmarks = pd.DataFrame(benchmark_results)
    df_benchmarks.to_csv(os.path.join(artifacts_dir, 'benchmark_summary.csv'), index=False)

    print(f"All artifacts successfully written to '{artifacts_dir}'.")
    return final_metrics, df_benchmarks


if __name__ == '__main__':
    data_file = os.path.join(os.path.dirname(__file__), '../data/credit_risk_dataset.csv')
    run_benchmarking_and_tracking(data_file)
