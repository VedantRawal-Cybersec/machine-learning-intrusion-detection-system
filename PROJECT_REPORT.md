# Project Report: Machine-Learning Intrusion Detection System

## Problem statement
Traditional signature-based controls may miss unusual traffic patterns. This project explores a supervised learning approach that classifies summarized network flows as normal or suspicious.

## Approach
A reproducible synthetic dataset was created because no private or production network traffic should be included in a public internship repository. The dataset contains eight interpretable flow features. A stratified train/test split was used, followed by numeric scaling and a class-balanced Random Forest classifier.

## Validation
The repository includes automated tests for dataset generation, training, model persistence, report generation, and inference. Metrics are exported to JSON and a confusion matrix is saved as an image.

## Result
On the bundled synthetic dataset, the model achieves high classification performance because the simulated attack patterns are intentionally separable. This result should not be presented as proof of production readiness. Real-world validation would require an approved benchmark dataset and monitoring for concept drift.

## Learning outcomes
- Feature engineering for network-flow data
- Supervised classification
- Precision, recall, F1 score, and confusion-matrix interpretation
- Reproducible model pipelines
- Defensive limitations and responsible reporting
