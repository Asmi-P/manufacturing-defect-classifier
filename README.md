# Manufacturing Defect Classification

**GitHub Repository:** https://github.com/Asmi-P/manufacturing-defect-classifier

**Live Streamlit Application:** https://manufacturing-defect-classifier-db7tlepol2rvmfbtivempe.streamlit.app/

## a. Problem Statement

The objective of this project is to implement, evaluate, and compare five machine-learning classification algorithms using a manufacturing defect dataset. The models are evaluated using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also includes an interactive Streamlit application that allows users to upload labelled test data, select a trained classifier, and view its evaluation metrics, confusion matrix, and classification report. The application loads saved models and evaluates the uploaded data without retraining.

## b. Dataset Description

- **Dataset:** Predicting Manufacturing Defects
- **Source:** Kaggle
- **Dataset link:** https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
- **Problem type:** Binary classification
- **Number of instances:** 3,240
- **Number of input features:** 16
- **Target column:** `DefectStatus`
- **Target labels:** 0 and 1

The input features cover production volume and cost, supplier quality, delivery delays, defect rate, quality score, maintenance hours, downtime, inventory turnover, stockout rate, worker productivity, safety incidents, energy consumption, energy efficiency, and additive manufacturing measures.

The target classes are imbalanced:

- **Class 0:** 517 records (approximately 15.96%)
- **Class 1:** 2,723 records (approximately 84.04%)

The dataset is synthetic. Therefore, the results from this experiment should not be interpreted as evidence of equivalent performance on real manufacturing operations.

## c. GitHub Repository Link

**Repository URL:** https://github.com/Asmi-P/manufacturing-defect-classifier

The repository contains the source code, trained models, test dataset, project notebook, dependencies, and documentation required to run and evaluate the application.

## d. Models Used and Evaluation

### Experimental Setup

All five classifiers were trained and evaluated using the same dataset and a stratified 80:20 train-test split.

- **Training records:** 2,592
- **Test records:** 648
- **Test split:** 20%
- **Random state:** 42
- **Stratification:** Applied to preserve the target-class proportions.

Precision, Recall, and F1 Score are reported for class 1. AUC is calculated using the predicted probability for class 1.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8765 | 0.7960 | 0.8921 | 0.9706 | 0.9297 | 0.4582 |
| Decision Tree | 0.8843 | 0.8617 | 0.9468 | 0.9138 | 0.9300 | 0.6004 |
| K-Nearest Neighbors (KNN) | 0.8673 | 0.7464 | 0.8857 | 0.9670 | 0.9246 | 0.4093 |
| Gaussian Naive Bayes | 0.8951 | 0.8268 | 0.9008 | 0.9835 | 0.9404 | 0.5479 |
| Support Vector Machine (SVM) | 0.8256 | 0.8157 | 0.9408 | 0.8459 | 0.8908 | 0.4805 |

### Observations on Model Performance

**Logistic Regression**

Achieved 87.65% accuracy and 97.06% recall for class 1. However, its MCC of 0.4582 and class-0 recall of approximately 37.86% indicate weaker recognition of class 0.

**Decision Tree**

Achieved the highest AUC (0.8617), precision (0.9468), and MCC (0.6004). Its class-0 recall was approximately 72.82%, indicating a more balanced result across the two classes than Logistic Regression, KNN, and Gaussian Naive Bayes in this experiment.

**K-Nearest Neighbors (KNN)**

Achieved 96.70% recall for class 1 but had the lowest AUC (0.7464) and MCC (0.4093). Its class-0 recall was approximately 33.98%, indicating weaker recognition of class 0.

**Gaussian Naive Bayes**

Achieved the highest accuracy (89.51%), recall (98.35%), and F1 Score (0.9404). However, its class-0 recall was approximately 42.72%, so its high overall scores should be interpreted alongside the class imbalance.

**Support Vector Machine (SVM)**

Achieved 94.08% precision for class 1 and a class-0 recall of approximately 71.84%. It had the lowest accuracy (82.56%) and F1 Score (0.8908) among the five models, reflecting a different trade-off between the two classes.

### Overall Model Comparison and Selection

Gaussian Naive Bayes is the strongest model when prioritising Accuracy, Recall, and F1 Score on this test split. Decision Tree performs best on AUC, Precision, and MCC.

If a single model must be selected based on the balance of the evaluated metrics, Decision Tree is a reasonable provisional choice because it achieved the highest MCC and AUC among the five models.

The final model selection depends on the relative cost of false positives and false negatives. In manufacturing, these costs should be considered before choosing a model for operational use.

### Important Limitations and Considerations

**Class imbalance:** Approximately 84.04% of the observations belong to class 1. Accuracy alone is therefore insufficient for model selection. Metrics such as MCC, AUC, class-specific recall, and the confusion matrix provide additional information.

**Feature sensitivity:** A sensitivity experiment excluding `DefectRate` and `QualityScore` produced substantially different results. This suggests that these features carry considerable predictive information, but it does not by itself prove data leakage. Their definitions and availability at prediction time should be verified before making real-world deployment claims.

**Generalisation:** The reported metrics are based on one held-out test split from a synthetic dataset. They do not guarantee equivalent performance on new data or real manufacturing operations.

## e. Streamlit Application

**Live application:** https://manufacturing-defect-classifier-db7tlepol2rvmfbtivempe.streamlit.app/

The Streamlit application provides the following functionality:

- Upload a CSV file containing labelled test data.
- Select one of the five trained classifiers using a dropdown.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the confusion matrix and classification report.
- Evaluate uploaded data using saved models without retraining.

The uploaded CSV must contain the required input features and the target column, `DefectStatus`, so that the application can evaluate predictions against the actual labels.

## f. Repository Contents

The GitHub repository is organised as follows:

- `app.py` — Streamlit application for model evaluation.
- `requirements.txt` — Python dependencies required to run the project.
- `test_data.csv` — labelled held-out test dataset used to evaluate the saved models.
- `model/` — directory containing the five saved classifiers:
  - `decision_tree.joblib`
  - `knn.joblib`
  - `logistic_regression.joblib`
  - `naive_bayes.joblib`
  - `svm.joblib`
- `assignment2.ipynb` — notebook containing the project workflow and experiments.
- `data/manufacturing_defect_dataset.csv` — original dataset used for the experiment.

## g. Reproducibility

The train-test split uses `test_size=0.20`, `random_state=42`, and stratification. The trained models are saved using Joblib and loaded by the Streamlit application.

The application evaluates uploaded labelled test data using the selected saved model. It does not retrain the model on uploaded data.

The reported metrics correspond to this experiment's held-out test split. Reproducing the same results requires the same dataset, preprocessing, model configurations, split, and evaluation procedures.

The dataset is synthetic, and performance on real-world manufacturing data may differ.

## h. Conclusion

This project compares five machine-learning classification algorithms for manufacturing defect classification using six evaluation metrics.

Gaussian Naive Bayes achieved the highest accuracy, recall, and F1 Score, while Decision Tree achieved the highest AUC and MCC. Considering the balance of metrics and performance across the two classes, Decision Tree is a reasonable provisional choice for this experiment.

The Streamlit application makes it possible to evaluate the saved classifiers interactively without retraining them. Further validation on representative real-world data, along with verification of feature availability and the cost of classification errors, would be required before operational deployment.
