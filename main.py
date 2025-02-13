import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle

# Set page config
st.set_page_config(page_title="Student Grade Prediction", page_icon="📚", layout="wide")

# Apply custom white background color theme
st.markdown("""
    <style>
        .reportview-container {
            background-color: #FFFFFF !important;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF !important;
        }
        .stSelectbox, .stNumberInput {
            max-width: 300px;
            margin-left: auto;
            margin-right: auto;
        }
        .st-expanderHeader {
            text-align: center;
        }
        .st-selectbox, .st-numberInput {
            display: flex;
            justify-content: center;
            margin-left: auto;
            margin-right: auto;
        }
        .stSelectbox > div {
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Student Grade Prediction System")
# Center the selectbox horizontally and place it at the top of the page
page = st.selectbox(
    "Select Prediction Type", ["Grade Prediction", "CGPA Prediction"], key="prediction_type", 
    help="Select whether you want to predict the grade or CGPA."
)

# Load the models and preprocessors
@st.cache_resource
def load_models():
    # You'll need to save these models using joblib or pickle first
    # For this example, we'll assume they're saved
    try:
        classification_model = joblib.load("best_classification_model.joblib")
        regression_model = joblib.load("best_regression_model.joblib")
        scaler = joblib.load("scaler.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        f_label_encoder = joblib.load("f_label_encoder.joblib")
        return (
            classification_model,
            regression_model,
            scaler,
            label_encoder,
            f_label_encoder,
        )
    except:
        st.error("Error loading models. Please ensure model files are present.")
        return None, None, None, None


# Create input fields
df = pd.read_csv("dataset.csv", index_col="ID")
df = df.drop(columns="Unnamed: 75")


# Remove target columns
classification_target = "Predicted Grade (based on CGPA) of 3rd semester"
regression_target = "PredictedCGPA of BS third sem."

# Get feature columns
feature_columns = [
    col for col in df.columns if col not in [classification_target, regression_target]
]

numeric_columns = (
    df[feature_columns].select_dtypes(include=["int64", "float64"]).columns
)
categorical_columns = df[feature_columns].select_dtypes(include=["object"]).columns


def create_feature_input():
    # Dynamically create input fields based on actual dataset columns
    features = {}

    # Create a scrollable vertical layout for feature inputs
    for col in feature_columns:
        with st.expander(f"Input for {col}"):
            if col in categorical_columns:
                unique_values = df[col].unique().tolist()
                features[col] = st.selectbox(
                    col,
                    unique_values,
                    key=col,
                    help="Select from the available options.",
                )
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                default_val = df[col].median()
                features[col] = st.number_input(
                    col,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    key=col,
                    help="Input the value for this feature."
                )

    return features


# Prediction function
def make_prediction(features, model_type="classification"):
    # Convert features to DataFrame

    # Preprocess the data
    classification_model, regression_model, scaler, label_encoder, f_label_encoder = (
        load_models()
    )
    processed_features = []
    for k, v in features.items():
        if k in categorical_columns:
            processed_features.append(f_label_encoder.transform([v])[0])
        else:
            processed_features.append(v)
    if model_type == "classification":
        prediction = classification_model.predict([processed_features])
        return label_encoder.inverse_transform(prediction)[0]
    else:
        prediction = regression_model.predict([processed_features])
        return round(prediction[0], 2)


# Main app logic
if page == "Grade Prediction":
    st.header("Grade Prediction")
    features = create_feature_input()

    if st.button("Predict Grade"):
        grade = make_prediction(features, "classification")

        # Display prediction with custom styling
        st.markdown(f"### Predicted Grade: {grade}", unsafe_allow_html=True)

else:  # CGPA Prediction
    st.header("CGPA Prediction")
    features = create_feature_input()

    if st.button("Predict CGPA"):
        cgpa = make_prediction(features, "regression")

        # Display prediction with custom styling
        st.markdown(f"### Predicted CGPA: {cgpa}", unsafe_allow_html=True)
