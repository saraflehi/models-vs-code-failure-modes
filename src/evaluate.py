from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baseline_rule import classify_log_entry


@dataclass(frozen=True)
class EvaluationRecord:
    timestamp: str
    service: str
    level: str
    message: str
    actual_label: int
    heuristic_prediction: int
    heuristic_risk_score: int
    statistical_prediction: int


def load_log_dataset(filepath: str) -> pd.DataFrame:
    """Load log data from CSV file."""
    return pd.read_csv(filepath)


def build_statistical_classifier(
    features: pd.Series, targets: pd.Series
) -> Tuple[Pipeline, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build and train ML classification pipeline with train/test split."""
    classifier = Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.3, random_state=42, stratify=targets
    )

    classifier.fit(X_train, y_train)
    return classifier, X_train, X_test, y_train, y_test


def run_comparative_evaluation(
    dataset: pd.DataFrame, statistical_model: Pipeline
) -> List[EvaluationRecord]:
    """Run both rule-based and ML predictions on dataset."""
    evaluation_records: List[EvaluationRecord] = []

    for _, entry in dataset.iterrows():
        severity = str(entry["level"])
        log_message = str(entry["message"])
        actual_label = int(entry["label"])

        # Get rule-based prediction
        heuristic_pred, risk_score, _detections = classify_log_entry(log_message, severity)

        # Get ML prediction
        combined_input = f"{severity} {log_message}"
        statistical_pred = int(statistical_model.predict([combined_input])[0])

        evaluation_records.append(
            EvaluationRecord(
                timestamp=str(entry["timestamp"]),
                service=str(entry["service"]),
                level=severity,
                message=log_message,
                actual_label=actual_label,
                heuristic_prediction=int(heuristic_pred),
                heuristic_risk_score=int(risk_score),
                statistical_prediction=int(statistical_pred),
            )
        )

    return evaluation_records


def export_evaluation_results(records: List[EvaluationRecord], output_path: str) -> None:
    """Save evaluation results to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "timestamp",
                "service",
                "level",
                "message",
                "actual_label",
                "heuristic_prediction",
                "heuristic_risk_score",
                "statistical_prediction",
            ]
        )
        for record in records:
            csv_writer.writerow(
                [
                    record.timestamp,
                    record.service,
                    record.level,
                    record.message,
                    record.actual_label,
                    record.heuristic_prediction,
                    record.heuristic_risk_score,
                    record.statistical_prediction,
                ]
            )


def main() -> None:
    """Main execution function for model comparison."""
    dataset = load_log_dataset("data/logs.csv")

    # Prepare features and train statistical model
    combined_features = dataset["level"] + " " + dataset["message"]
    target_labels = dataset["label"]

    statistical_model, _X_train, _X_test, _y_train, _y_test = build_statistical_classifier(
        combined_features, target_labels
    )

    # Generate predictions on entire dataset for performance metrics
    heuristic_predictions: List[int] = []
    statistical_predictions: List[int] = []
    ground_truth: List[int] = []

    for _, entry in dataset.iterrows():
        severity = str(entry["level"])
        log_message = str(entry["message"])
        actual_label = int(entry["label"])

        heuristic_pred, _risk_score, _ = classify_log_entry(log_message, severity)
        statistical_pred = int(statistical_model.predict([f"{severity} {log_message}"])[0])

        heuristic_predictions.append(int(heuristic_pred))
        statistical_predictions.append(int(statistical_pred))
        ground_truth.append(actual_label)

    print("=== HEURISTIC-BASED CLASSIFIER PERFORMANCE (Full Dataset) ===")
    print(classification_report(ground_truth, heuristic_predictions))
    print("Confusion Matrix (Heuristic):")
    print(confusion_matrix(ground_truth, heuristic_predictions))

    print("\n=== STATISTICAL CLASSIFIER PERFORMANCE (Full Dataset) ===")
    print(classification_report(ground_truth, statistical_predictions))
    print("Confusion Matrix (Statistical):")
    print(confusion_matrix(ground_truth, statistical_predictions))

    evaluation_records = run_comparative_evaluation(dataset, statistical_model)
    export_evaluation_results(evaluation_records, "reports/comparison.csv")

    print("\nComparison results exported to: reports/comparison.csv")


if __name__ == "__main__":
    main()
