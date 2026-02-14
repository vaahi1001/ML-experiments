
#/mount/src/ml-experiments/01_ClassificationModels/
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
import numpy as np


st.title("Heart Disease Detection")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

# Sex
sex_option = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_option == "Female" else 1

# Chest Pain Type
cp_option = st.selectbox("Chest Pain Type",
                         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
cp = cp_map[cp_option]

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)

# Fasting Blood Sugar
fbs_option = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 0 if fbs_option == "No" else 1

# Resting ECG
restecg_option = st.selectbox("Resting ECG",
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_map[restecg_option]

thalach = st.number_input("Max Heart Rate Achieved", value=150)

# Exercise Induced Angina
exang_option = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 0 if exang_option == "No" else 1

oldpeak = st.number_input("ST Depression", value=1.0, step=0.1)

# Slope of ST segment
slope_option = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_map[slope_option]

# Number of major vessels
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# Thalassemia
thal_option = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_map[thal_option]

# Prepare input DataFrame
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


# --- Create a readable version for display ---
gender_map = {0: "Female", 1: "Male"}
cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
fbs_map = {0: "No", 1: "Yes"}
restecg_map = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
exang_map = {0: "No", 1: "Yes"}
slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_map = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

patient_readable = input_data.copy()
patient_readable['sex'] = patient_readable['sex'].map(gender_map)
patient_readable['cp'] = patient_readable['cp'].map(cp_map)
patient_readable['fbs'] = patient_readable['fbs'].map(fbs_map)
patient_readable['restecg'] = patient_readable['restecg'].map(restecg_map)
patient_readable['exang'] = patient_readable['exang'].map(exang_map)
patient_readable['slope'] = patient_readable['slope'].map(slope_map)
patient_readable['thal'] = patient_readable['thal'].map(thal_map)

st.write("### Patient Details (Readable Format)")
st.dataframe(patient_readable)

# --- Load all model PKLs dynamically ---
model_files = [
    "/mount/src/ml-experiments/01_ClassificationModels/logisticregression_pipeline.pkl",
    "/mount/src/ml-experiments/01_ClassificationModels/decisiontree_pipeline.pkl",
    "/mount/src/ml-experiments/01_ClassificationModels/k-nearestneighbor_pipeline.pkl",
    "/mount/src/ml-experiments/01_ClassificationModels/naivebayes_pipeline.pkl",
    "/mount/src/ml-experiments/01_ClassificationModels/randomforest_pipeline.pkl",
    "/mount/src/ml-experiments/01_ClassificationModels/xgboost_pipeline.pkl"
]

models = {}
for file in model_files:
    if os.path.exists(file):
        model_name = os.path.basename(file).replace("_pipeline.pkl", "").replace("kneighbors", "K-NearestNeighbor").capitalize()
        models[model_name] = joblib.load(file)
    else:
        st.warning(f"Model file {file} not found!")

# --- Button to run predictions ---
if st.button("Predict with all models"):

    if not models:
        st.error("No models loaded. Check PKL files.")
    else:
        # --- 1️⃣ Predictions for user input ---
        results = []
        for name, model in models.items():
            try:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][pred]
                risk = "High Risk" if pred == 1 else "Low Risk"
            except Exception as e:
                pred, prob, risk = "Error", "Error", f"Error: {e}"
         # Get probability safely
            try:
                prob_array = model.predict_proba(input_data)
                # prob_array could be 2D [[0.1, 0.9]] or 1D [0.9]
                if prob_array.ndim == 2:
                    prob_value = float(prob_array[0][1])  # second column = 'High Risk'
                else:
                    prob_value = float(prob_array[0])
            except AttributeError:
                # model has no predict_proba
                prob_value = None
            
            # Determine risk label
            risk = "High Risk" if prob_value is not None and prob_value > 0.5 else "Low Risk"
            
            # Append safely
            results.append({
                "Model": name,
                "Prediction": risk,
                "Probability": f"{prob_value:.2f}" if prob_value is not None else "N/A"
            })



        st.write("### Predictions for this patient")
        st.table(pd.DataFrame(results))

        # --- 2️⃣ KPIs on test data ---
        metrics_list = []
        for name, model in models.items():
            try:
                X_test = model.X_test
                y_test = model.y_test
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:,1]
                metrics_list.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 2),
                    "AUC": round(roc_auc_score(y_test, y_proba), 2),
                    "Precision": round(precision_score(y_test, y_pred), 2),
                    "Recall": round(recall_score(y_test, y_pred), 2),
                    "F1 Score": round(f1_score(y_test, y_pred), 2),
                    "MCC": round(matthews_corrcoef(y_test, y_pred), 2)
                })
            except Exception as e:
                metrics_list.append({
                    "Model": name,
                    "Accuracy": "Error",
                    "AUC": "Error",
                    "Precision": "Error",
                    "Recall": "Error",
                    "F1 Score": "Error",
                    "MCC": "Error"
                })

        st.write("### Model KPIs on Test Set")
        st.table(pd.DataFrame(metrics_list))

        # Try loading
        try:
            xgb_model = joblib.load('xgboost_model.pkl')
            print("Model loaded successfully")
        except Exception as e:
            print("Error loading model:", e)
        
        # Try predicting on test data
        try:
            preds = xgb_model.predict(X_test)
            probs = xgb_model.predict_proba(X_test)
            print(preds[:5], probs[:5])
        except Exception as e:
            print("Error predicting:", e)


