# Manufacturing Defect Classification

## a. Problem Statement

The objective of this project is to implement and compare five machine-learning classification algorithms on a manufacturing defect dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

A Streamlit web application allows users to upload labelled test data, select a trained model and view its evaluation metrics, confusion matrix and classification report. The application loads saved models and evaluates the uploaded data without retraining.

## b. Dataset Description

- **Dataset:** Predicting Manufacturing Defects
- **Source:** Kaggle
- **Dataset link:** https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
- **Problem type:** Binary classification
- **Number of instances:** 3,240
- **Number of input features:** 16
- **Target column:** `DefectStatus`
- **Target labels:** 0 and 1

The input features cover production volume and cost, supplier quality, delivery delays, defect rate, quality score, maintenance hours, downtime, inventory, worker productivity, safety incidents, energy use and additive manufacturing measures.

The dataset is synthetic. Results from this experiment should not be interpreted as evidence of equivalent performance on real manufacturing operations.

## c. GitHub Repository Link

**Repository URL:** TODO — add the published GitHub repository URL.

## d. Models Used and Evaluation

All five classification models were trained and evaluated using the same dataset and stratified 80:20 train-test split. The test set contains 648 records.

Precision, Recall and F1 Score are reported for class 1. AUC is calculated using the predicted probability for class 1.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8765 | 0.7960 | 0.8921 | 0.9706 | 0.9297 | 0.4582 |
| Decision Tree | 0.8843 | 0.8617 | 0.9468 | 0.9138 | 0.9300 | 0.6004 |
| KNN | 0.8673 | 0.7464 | 0.8857 | 0.9670 | 0.9246 | 0.4093 |
| Gaussian Naive Bayes | 0.8951 | 0.8268 | 0.9008 | 0.9835 | 0.9404 | 0.5479 |
| Support Vector Machine | 0.8256 | 0.8157 | 0.9408 | 0.8459 | 0.8908 | 0.4805 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 87.65% accuracy and 97.06% recall for class 1. However, its MCC of 0.4582 and class-0 recall of approximately 37.86% indicate difficulty identifying class 0. |
| Decision Tree | Achieved the highest AUC (0.8617), precision (0.9468) and MCC (0.6004). Its class-0 recall was approximately 72.82%, making it more balanced across the two classes than Logistic Regression, KNN and Naive Bayes in this experiment. |
| KNN | Achieved 96.70% recall for class 1 but had the lowest AUC (0.7464) and MCC (0.4093). Its class-0 recall was approximately 33.98%, indicating weaker recognition of class 0. |
| Gaussian Naive Bayes | Achieved the highest accuracy (89.51%), recall (98.35%) and F1 Score (0.9404). However, its class-0 recall was approximately 42.72%, so its high overall scores should be interpreted alongside the class imbalance. |
| Support Vector Machine | Achieved 94.08% precision for class 1, but had the lowest accuracy (82.56%) and F1 Score (0.8908). Its class-0 recall was approximately 71.84%, reflecting a different trade-off between the two classes. |

### Overall Winner for This Dataset

Gaussian Naive Bayes is the winner when prioritising Accuracy, Recall and F1 Score. Decision Tree is the winner when prioritising AUC, Precision and MCC.

If a single model must be selected using the balance of metrics and performance across both classes, Decision Tree is a reasonable provisional choice because it has the highest MCC and AUC among the five models. The final choice depends on the relative cost of errors for each class.

### Important Limitation

The target classes are imbalanced, with approximately 84% of the observations in class 1. Accuracy alone is therefore insufficient for selecting a model.

A sensitivity experiment excluding `DefectRate` and `QualityScore` produced substantially different results. This suggests that these features carry considerable predictive information, but does not by itself prove data leakage. Their definitions and availability at prediction time should be verified before making real-world deployment claims.

## Streamlit Application

The application provides:

- CSV upload for labelled test data
- A dropdown to select one of the five saved classifiers
- Accuracy, AUC, Precision, Recall, F1 Score and MCC
- Confusion matrix and classification report
- Evaluation using saved models without retraining

**Live Streamlit App:** TODO — add the deployed Streamlit Community Cloud URL.

## Repository Contents

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `test_data.csv` — labelled held-out test dataset
- `model/` — saved trained models
- `notebooks/` — project notebook and experiments

## Reproducibility

The train-test split uses `test_size=0.20`, `random_state=42` and stratification. The trained models are saved using Joblib and loaded by the Streamlit application.

The reported metrics are from this experiment's held-out test split. They do not guarantee the same performance on new or real-world data.
