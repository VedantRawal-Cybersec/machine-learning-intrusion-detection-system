from src.data_generator import FEATURE_COLUMNS, generate_dataset


def test_generated_dataset_shape_and_columns() -> None:
    data = generate_dataset(rows=300, seed=7)
    assert len(data) == 300
    assert set(FEATURE_COLUMNS + ["label"]) == set(data.columns)
    assert set(data["label"].unique()) == {"normal", "attack"}
