import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

# File paths
dataset_file = '2020/heart_2020_cleaned.csv'
model_file = "random_forest_model.joblib"

# Load model and encoders
final_model = joblib.load(model_file)
df = pd.read_csv(dataset_file)

# Load saved label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load the original encoded column structure
with open('encoded_columns.pkl', 'rb') as f:
    encoded_columns = pickle.load(f)

# List of binary columns that were label-encoded
binary_cols = [
    "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Sex", "Diabetic", "PhysicalActivity", "GenHealth", "Asthma",
    "KidneyDisease", "SkinCancer"
]

# Streamlit App Title
st.title("Health Risk Predictor 🏥")
st.write("Enter the feature values below to predict the Heart Disease:")

# Create three columns
col1, col2, col3 = st.columns(3)

# First column inputs
with col1:
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, step=0.1)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Drinking", ["Yes", "No"])
    stroke = st.selectbox("Stroke", ["Yes", "No"])
    physical_health = st.slider("Physical Health (Days)", 0, 30, 0)
    mental_health = st.slider("Mental Health (Days)", 0, 30, 0)
    diff_walking = st.selectbox("Difficulty Walking", ["Yes", "No"])
    sex = st.selectbox("Sex", ["Male", "Female"])

# Spacer column (optional)
with col2:
    st.empty()

# Third column inputs
with col3:
    age_category = st.selectbox("Age Category", [
        '65-69', '60-64', '70-74', '55-59', '50-54', '80 or older',
        '45-49', '75-79', '18-24', '40-44', '35-39', '30-34', '25-29'
    ])
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
        'Hispanic', 'Other'
    ])
    diabetic = st.selectbox("Diabetic", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
    physical_activity = st.selectbox("Physical Activity", ["Yes", "No"])
    gen_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
    sleep_time = st.slider("Sleep Time (hours)", 0, 24, 7)
    asthma = st.selectbox("Asthma", ["Yes", "No"])
    kidney_disease = st.selectbox("Kidney Disease", ["Yes", "No"])
    skin_cancer = st.selectbox("Skin Cancer", ["Yes", "No"])

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholDrinking": alcohol,
    "Stroke": stroke,
    "PhysicalHealth": physical_health,
    "MentalHealth": mental_health,
    "DiffWalking": diff_walking,
    "Sex": sex,
    "AgeCategory": age_category,
    "Race": race,
    "Diabetic": diabetic,
    "PhysicalActivity": physical_activity,
    "GenHealth": gen_health,
    "SleepTime": sleep_time,
    "Asthma": asthma,
    "KidneyDisease": kidney_disease,
    "SkinCancer": skin_cancer
}])

# Apply label encoding for binary columns
for col in binary_cols:
    if col in input_data.columns:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

# Apply one-hot encoding for multi-class columns
input_data = pd.get_dummies(input_data, columns=["AgeCategory", "Race"], drop_first=True)

# Ensure the new dataframe has the same structure as the training data
for col in encoded_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing column
input_data = input_data[encoded_columns]  # Reorder columns

# Remove the target column ('HeartDisease') from the input_data (not needed during prediction)
input_data = input_data.drop(columns=["HeartDisease"], errors='ignore')  # In case 'HeartDisease' exists in input_data

# Single Predict Button with unique key
if st.button("Predict Heart Disease Risk", key="predict_btn"):
    prediction = final_model.predict(input_data)

    if prediction[0] == 1:
        st.write("💔 The prediction indicates that there **is a risk** of Heart Disease.")
    else:
        st.write("💖 The prediction indicates that there is **no risk** of Heart Disease.")
