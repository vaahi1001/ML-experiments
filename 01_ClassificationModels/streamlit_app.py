
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


st.title("Heart Disease Detection - Multi-Model Demo")

# --- User Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1])
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
            results.append({"Model": name, "Prediction": risk, "Probability": f"{prob:.2f}"})

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

