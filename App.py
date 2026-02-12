# ==========================================================
# DRY BEAN CLASSIFICATION - STREAMLIT APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("🌱 Dry Bean Classification App")
st.markdown("Upload test dataset and evaluate selected ML model.")

# ----------------------------------------------------------
# Download Sample Test Dataset
# ----------------------------------------------------------

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as file:
        st.download_button(
            label="Download Sample Test Dataset",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )

# ----------------------------------------------------------
# Model Selection
# ----------------------------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_options = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "saved_models", "logistic_regression.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "model", "saved_models", "decision_tree.pkl"),
    "KNN": os.path.join(BASE_DIR, "model", "saved_models", "knn.pkl"),
    "Naive Bayes": os.path.join(BASE_DIR, "model", "saved_models", "naive_bayes.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "model", "saved_models", "random_forest.pkl"),
    "XGBoost": os.path.join(BASE_DIR, "model", "saved_models", "xgboost.pkl")
}

selected_model_name = st.selectbox("Select Model", list(model_options.keys()))

# ----------------------------------------------------------
# Dataset Upload
# ----------------------------------------------------------

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    if "Class" not in df.columns:
        st.error("Uploaded file must contain 'Class' column as target.")
    else:
        from sklearn.preprocessing import LabelEncoder
        X = df.drop("Class", axis=1)
        y = df["Class"]

# Encode class labels (same as training logic)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        # ------------------------------------------------------
        # Load Selected Model
        # ------------------------------------------------------

        model_path = model_options[selected_model_name]

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # ------------------------------------------------------
        # Make Predictions
        # ------------------------------------------------------

        y_pred = model.predict(X)

        try:
            y_proba = model.predict_proba(X)
            auc = roc_auc_score(y, y_proba, multi_class="ovr")
        except:
            auc = "Not Available"

        # ------------------------------------------------------
        # Calculate Metrics
        # ------------------------------------------------------

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted")
        recall = recall_score(y, y_pred, average="weighted")
        f1 = f1_score(y, y_pred, average="weighted")
        mcc = matthews_corrcoef(y, y_pred)

        # ------------------------------------------------------
        # Display Metrics
        # ------------------------------------------------------

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(acc, 4))
        col2.metric("AUC Score", auc if isinstance(auc, str) else round(auc, 4))
        col3.metric("Precision", round(precision, 4))

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", round(recall, 4))
        col5.metric("F1 Score", round(f1, 4))
        col6.metric("MCC Score", round(mcc, 4))

        # ------------------------------------------------------
        # Confusion Matrix
        # ------------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        st.write(cm)

        # ------------------------------------------------------
        # Classification Report
        # ------------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

else:
    st.info("Please upload a test CSV file to evaluate the model.")
