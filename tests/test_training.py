from pathlib import Path

from src.data_generator import generate_dataset
from src.predict import predict_record
from src.train import train_model


def test_training_and_prediction(tmp_path: Path) -> None:
    data_path = tmp_path / "flows.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "reports"
    generate_dataset(rows=700, seed=21).to_csv(data_path, index=False)

    metrics = train_model(data_path, model_path, report_path)
    assert model_path.exists()
    assert metrics["accuracy"] >= 0.80

    result = predict_record(
        model_path,
        {
            "duration_seconds": 0.2,
            "source_bytes": 150.0,
            "destination_bytes": 20.0,
            "packet_count": 95,
            "failed_logins": 6,
            "same_service_rate": 0.15,
            "syn_flag_count": 18,
            "unique_destination_ports": 24,
        },
    )
    assert result["prediction"] in {"normal", "attack"}
