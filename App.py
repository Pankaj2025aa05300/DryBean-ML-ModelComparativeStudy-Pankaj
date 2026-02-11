import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Dry Bean Classification",
    layout="wide"
)

st.title("Dry Bean Classification – ML Model Comparison")
st.write("Upload test data and select a trained model to evaluate performance.")

# ----------------------------------
# Load Models & Encoder
# ----------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("models/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("models/xgboost.pkl", "rb")),
    }
    label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
    return models, label_encoder

models, label_encoder = load_models()

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("User Input")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ----------------------------------
# Main Logic
# ----------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Class" not in df.columns:
        st.error("❌ Uploaded dataset must contain 'Class' column")
    else:
        X_test = df.drop("Class", axis=1)
        y_test = df["Class"]

        model = models[model_name]
        y_pred = model.predict(X_test)

        # ----------------------------------
        # Metrics
        # ----------------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        col3.metric("MCC", f"{mcc:.4f}")

        # ----------------------------------
        # Confusion Matrix
        # ----------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # ----------------------------------
        # Classification Report
        # ----------------------------------
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
