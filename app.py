
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Manufacturing Defect Classifier",
    page_icon="🏭",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}

FEATURE_COLUMNS = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "DefectRate",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost",
]
TARGET_COLUMN = "DefectStatus"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("🏭 Manufacturing Defect Classification")
st.write(
    "Evaluate five trained machine-learning classifiers "
    "using labelled test data."
)
st.caption(
    "The app loads saved models and evaluates the uploaded CSV. "
    "It does not retrain models."
)

st.info(
    "Upload a CSV containing all 16 input features and "
    "the DefectStatus target column."
)

uploaded_file = st.file_uploader(
    "Upload labelled test data",
    type=["csv"],
)

selected_model_name = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys()),
)

if uploaded_file is None:
    st.caption("Use test_data.csv from your project folder to begin.")
else:
    try:
        test_df = pd.read_csv(uploaded_file)

        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [
            column for column in required_columns
            if column not in test_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        # Select only required columns in the expected order.
        X_uploaded = test_df[FEATURE_COLUMNS].copy()
        y_true = test_df[TARGET_COLUMN]

        # Check for missing and non-numeric feature values.
        X_uploaded = X_uploaded.apply(
            pd.to_numeric, errors="coerce"
        )
        y_numeric = pd.to_numeric(y_true, errors="coerce")

        if X_uploaded.isnull().any().any() or y_numeric.isnull().any():
            st.error(
                "Missing or non-numeric values were found "
                "in the required columns. Please clean the CSV."
            )
            st.stop()

        if not y_numeric.isin([0, 1]).all():
            st.error(
                "DefectStatus must contain only the labels 0 and 1."
            )
            st.stop()

        y_true = y_numeric.astype(int)

        if y_true.nunique() != 2:
            st.error(
                "AUC requires both target classes (0 and 1) "
                "to be present in the uploaded test data."
            )
            st.stop()

        model_path = MODEL_DIR / MODEL_FILES[selected_model_name]

        if not model_path.exists():
            st.error(
                f"Saved model not found: {model_path.name}. "
                "Check the model folder."
            )
            st.stop()

        model = load_model(str(model_path))

        y_pred = model.predict(X_uploaded)
        probabilities = model.predict_proba(X_uploaded)

        # Locate class 1 explicitly for the AUC calculation.
        class_labels = list(model.classes_)
        positive_class_index = class_labels.index(1)
        y_prob = probabilities[:, positive_class_index]

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        st.subheader(f"Evaluation: {selected_model_name}")
        st.write(f"**Test records:** {len(test_df):,}")
        st.caption(
            "Precision, recall and F1 use class 1 as the positive class."
        )

        metric_columns = st.columns(3)
        for column, metric_name in zip(
            metric_columns, ["Accuracy", "AUC", "Precision"]
        ):
            column.metric(metric_name, f"{metrics[metric_name]:.4f}")

        metric_columns = st.columns(3)
        for column, metric_name in zip(
            metric_columns, ["Recall", "F1 Score", "MCC"]
        ):
            column.metric(metric_name, f"{metrics[metric_name]:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Classification Report")
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            "The file could not be evaluated. Check its format "
            f"and the saved models. Details: {exc}"
        )
