"""Train and evaluate the intrusion-detection model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generator import FEATURE_COLUMNS


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(data_path: Path, model_path: Path, report_dir: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    missing = set(FEATURE_COLUMNS + ["label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    x = data[FEATURE_COLUMNS]
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_attack": round(
            float(precision_score(y_test, predictions, pos_label="attack")), 4
        ),
        "recall_attack": round(
            float(recall_score(y_test, predictions, pos_label="attack")), 4
        ),
        "f1_attack": round(float(f1_score(y_test, predictions, pos_label="attack")), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    details = classification_report(y_test, predictions, output_dict=True)
    (report_dir / "metrics.json").write_text(
        json.dumps({"summary": metrics, "classification_report": details}, indent=2),
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["normal", "attack"], values_format="d"
    )
    display.ax_.set_title("IDS Confusion Matrix")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML IDS model")
    parser.add_argument("--data", type=Path, default=Path("data/network_flows.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/ids_model.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    metrics = train_model(args.data, args.model, args.reports)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
