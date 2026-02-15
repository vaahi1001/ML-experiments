
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

st.write("Summary of Patient Details : ")
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
    try:
        name = os.path.basename(file).split("_pipeline")[0].capitalize()
        models[name] = joblib.load(file)
    except Exception as e:
        st.warning(f"Could not load {file}: {e}")

# --- Model Selection Section ---
st.markdown("## Select Models for Prediction")
model_names = list(models.keys())
# Option to select all
select_all = st.checkbox("Select All Models")
if select_all:
    selected_models = model_names
else:
    selected_models = st.multiselect(
        "Choose Models:",
        model_names
    )

if st.button("Run Prediction"):
    if not selected_models:
        st.warning("Please select at least one model.")
    
    else:
        prediction_results = []
        feature_results = []

        for name in selected_models:
            model = models[name]

            try:
                pred = model.predict(input_data)[0]

                try:
                    prob_array = model.predict_proba(input_data)
                    prob_value = float(prob_array[0][1])
                except:
                    prob_value = None

                risk = "High Risk" if pred == 1 else "Low Risk"

                prediction_results.append({
                    "Model": name,
                    "Prediction": risk,
                    "Probability (%)": f"{prob_value*100:.2f}%" if prob_value is not None else "N/A"
                })

           
                # -----------------------------
                # Sorted Feature Importance
                #-----------------------------
                if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Importance": importance
                }).sort_values(by="Importance", ascending=False)
                feature_df["Importance"] = feature_df["Importance"].round(3)
                feature_df["Model"] = name
                feature_results.extend(feature_df.to_dict(orient="records"))


            except Exception as e:
                prediction_results.append({
                    "Model": name,
                    "Prediction": f"Error: {e}",
                    "Probability (%)": "Error"
                })

        st.markdown("### Prediction Results")
        st.table(pd.DataFrame(prediction_results))

        if feature_results:
            st.markdown("### Model Feature Importance")
            st.dataframe(pd.DataFrame(feature_results))

### KPI details should also be dynamic
      
        st.markdown("### Model KPIs (Test Set)")

        kpi_results = []

        for name in selected_models:
            model = models[name]

            try:
                X_test = model.X_test
                y_test = model.y_test

                y_pred = model.predict(X_test)

                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc_score = round(roc_auc_score(y_test, y_proba), 2)
                except:
                    auc_score = "N/A"

                kpi_results.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 2),
                    "AUC": auc_score,
                    "Precision": round(precision_score(y_test, y_pred), 2),
                    "Recall": round(recall_score(y_test, y_pred), 2),
                    "F1 Score": round(f1_score(y_test, y_pred), 2),
                    "MCC": round(matthews_corrcoef(y_test, y_pred), 2)
                })

            except Exception as e:
                kpi_results.append({
                    "Model": name,
                    "Accuracy": "Unavailable",
                    "AUC": "Unavailable",
                    "Precision": "Unavailable",
                    "Recall": "Unavailable",
                    "F1 Score": "Unavailable",
                    "MCC": "Unavailable"
                })

        if kpi_results:
            st.table(pd.DataFrame(kpi_results))




