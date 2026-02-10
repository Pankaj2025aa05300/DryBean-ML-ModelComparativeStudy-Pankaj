import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("🌱 Dry Bean Classification App")
st.write("Upload the Dry Bean dataset and train multiple ML models.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Dry Bean Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # -----------------------------
    # Load dataset
    # -----------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Basic cleaning
    # -----------------------------
    df = df.dropna()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "Class" not in df.columns:
        st.error("❌ Dataset must contain a 'Class' column")
        st.stop()

    # -----------------------------
    # Features & target
    # -----------------------------
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Model selection
    # -----------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
        ),
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)

    # -----------------------------
    # Train model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {acc:.4f}")

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("⬆️ Please upload the Dry Bean dataset to begin.")
