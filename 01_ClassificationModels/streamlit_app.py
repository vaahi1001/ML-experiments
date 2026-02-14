import streamlit as st
import joblib
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

st.write("Hello World")  # Works inside Streamlit app
print("Hello World")     # Works in terminal, but might not show in Streamlit UI

pipeline = joblib.load("/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl")


st.title("Heart Disease Prediction App")

st.write("""
Enter patient details below to predict heart disease:
""")

# Step 2: User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Step 3: Prepare input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Step 4: Make prediction
if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][prediction]
    if prediction == 1:
        st.error(f"High risk of heart disease! (Probability: {probability:.2f})")
    else:
        st.success(f"Low risk of heart disease. (Probability: {probability:.2f})")
