import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load models safely (NO caching, NO top-level loading)
# -----------------------------
def load_models():
    models = {}

    with open("model/random_forest.pkl", "rb") as f:
        models["Random Forest"] = pickle.load(f)

    with open("model/logistic_regression.pkl", "rb") as f:
        models["Logistic Regression"] = pickle.load(f)

    with open("model/decision_tree.pkl", "rb") as f:
        models["Decision Tree"] = pickle.load(f)

    with open("model/knn.pkl", "rb") as f:
        models["KNN"] = pickle.load(f)

    with open("model/naive_bayes.pkl", "rb") as f:
        models["Naive Bayes"] = pickle.load(f)

    with open("model/xgboost.pkl", "rb") as f:
        models["XGBoost"] = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        models["Label Encoder"] = pickle.load(f)

    return models


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Dry Bean Classification", layout="wide")
    st.title("🌱 Dry Bean Classification App")

    st.write(
        """
        This application predicts the **class of Dry Beans**
        using multiple Machine Learning models.
        """
    )

    # Load models only when app starts
    models = load_models()
    label_encoder = models["Label Encoder"]

    # Feature names (from Dry Bean Dataset)
    feature_names = [
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
        "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
        "Extent", "Solidity", "roundness", "Compactness",
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
    ]

    st.sidebar.header("🔢 Input Features")

    # User inputs
    user_input = []
    for feature in feature_names:
        value = st.sidebar.number_input(
            label=feature,
            min_value=0.0,
            value=1.0,
            step=0.1
        )
        user_input.append(value)

    input_array = np.array(user_input).reshape(1, -1)

    # Model selection
    st.sidebar.header("🤖 Select Model")
    model_name = st.sidebar.selectbox(
        "Choose a model:",
        [
            "Random Forest",
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "XGBoost"
        ]
    )

    if st.button("🔍 Predict"):
        model = models[model_name]

        prediction_encoded = model.predict(input_array)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"### ✅ Predicted Dry Bean Class: **{prediction_label}**")

        # Show probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_array)[0]
            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)

            st.subheader("📊 Prediction Probabilities")
            st.dataframe(prob_df, use_container_width=True)


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    main()
