"""Score one network-flow record with the trained IDS model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_generator import FEATURE_COLUMNS


def predict_record(model_path: Path, record: dict[str, float]) -> dict[str, object]:
    missing = set(FEATURE_COLUMNS) - set(record)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    model = joblib.load(model_path)
    frame = pd.DataFrame([{column: record[column] for column in FEATURE_COLUMNS}])
    label = str(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    probability_map = {
        str(name): round(float(value), 4)
        for name, value in zip(model.classes_, probabilities, strict=True)
    }
    return {"prediction": label, "probabilities": probability_map}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a network-flow label")
    parser.add_argument("--model", type=Path, default=Path("models/ids_model.joblib"))
    parser.add_argument("--json", required=True, help="JSON object containing flow features")
    args = parser.parse_args()
    print(json.dumps(predict_record(args.model, json.loads(args.json)), indent=2))


if __name__ == "__main__":
    main()
