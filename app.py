# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("predictive_maintenance_model.pkl")

st.title("🔧 IoT Sensor Data Analyzer - Predictive Maintenance")

st.markdown("Provide sensor readings to predict machine failure.")

# Define feature input form
def user_input_features():
    st.subheader("Enter Sensor Values")

    # You can dynamically load from your training data's column names
    input_data = {}

    # Load training data to get column names
    df = pd.read_csv("eda_cleaned_sensor_data.csv")
    df.columns = df.columns.str.strip()

    # Drop target column and datetime if any
    drop_cols = ["failure"]
    drop_cols += [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]

    features = [col for col in df.columns if col not in drop_cols]

    for feature in features:
        if df[feature].dtype == "object":
            unique_vals = df[feature].unique()
            input_data[feature] = st.selectbox(f"{feature}", unique_vals)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))

    return pd.DataFrame([input_data])

# Get user input
input_df = user_input_features()

# Encode input the same way as during training
df_original = pd.read_csv("eda_cleaned_sensor_data.csv")
df_original.columns = df_original.columns.str.strip()

# Clean and encode training data columns to mimic model training
for col in input_df.columns:
    if df_original[col].dtype == "object":
        input_df[col] = input_df[col].astype("category")
        input_df[col] = input_df[col].cat.set_categories(df_original[col].astype("category").cat.categories)
        input_df[col] = input_df[col].cat.codes

# Predict
if st.button("🔍 Predict Failure"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Warning: Failure Predicted!")
    else:
        st.success("✅ No Failure Detected.")
