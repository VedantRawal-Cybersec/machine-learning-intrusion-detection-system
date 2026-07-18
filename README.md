# Machine-Learning Intrusion Detection System

A compact educational IDS that classifies network-flow records as **normal** or **attack**. The project demonstrates a complete defensive machine-learning workflow: dataset generation, preprocessing, model training, evaluation, artifact storage, and single-record prediction.

> **Ethical-use notice:** This project is designed for defensive education and simulated data. It does not capture private traffic, exploit systems, or modify production data.

## Project objectives

- Build a reproducible network-traffic dataset for experimentation.
- Train a supervised classifier using measurable traffic features.
- Evaluate the model with accuracy, precision, recall, F1 score, and a confusion matrix.
- Save the trained pipeline and reuse it for later predictions.

## Technology

- Python 3.10+
- pandas and NumPy
- scikit-learn
- Matplotlib
- pytest

## Repository structure

```text
ml_ids_project/
├── data/                  # Generated sample dataset
├── models/                # Trained model artifact
├── reports/               # Metrics and confusion matrix
├── src/
│   ├── data_generator.py
│   ├── train.py
│   └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

## Features used by the model

- Connection duration
- Source and destination byte counts
- Packet count
- Failed login count
- Same-service connection rate
- SYN flag count
- Number of unique destination ports

These features are intentionally understandable and suitable for explaining the project during an internship evaluation.

## Installation

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

Generate the dataset:

```bash
python -m src.data_generator --rows 4000
```

Train and evaluate the model:

```bash
python -m src.train
```

The command creates:

- `models/ids_model.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

Make a prediction:

```bash
python -m src.predict --json '{"duration_seconds":0.2,"source_bytes":150,"destination_bytes":20,"packet_count":95,"failed_logins":6,"same_service_rate":0.15,"syn_flag_count":18,"unique_destination_ports":24}'
```

Windows PowerShell example:

```powershell
python -m src.predict --json '{"duration_seconds":0.2,"source_bytes":150,"destination_bytes":20,"packet_count":95,"failed_logins":6,"same_service_rate":0.15,"syn_flag_count":18,"unique_destination_ports":24}'
```

## Run tests

```bash
pytest -q
```

## Methodology

1. A seeded generator creates labelled network-flow records.
2. Data is split into stratified training and testing sets.
3. Numeric features are standardized in a scikit-learn pipeline.
4. A class-balanced Random Forest learns the traffic patterns.
5. Performance is recorded in machine-readable JSON and a confusion matrix.

## Limitations and future work

- Synthetic traffic is useful for demonstration but does not represent every real attack.
- A production IDS would require approved packet/flow collection, stronger validation, concept-drift monitoring, and analyst review.
- Future versions could support CIC-IDS datasets, SHAP explanations, a dashboard, and streaming inference.

## Suggested LinkedIn/GitHub description

> Developed a machine-learning Intrusion Detection System in Python using scikit-learn. Built a reproducible network-flow dataset, trained a class-balanced Random Forest model, generated evaluation reports, and added reusable prediction and automated tests. The project is intended for defensive cybersecurity education.

## Author

Vedant Rawal — Cybersecurity and AI learner
