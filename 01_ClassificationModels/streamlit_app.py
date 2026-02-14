import streamlit as st
import joblib
import pandas as pd

st.title("Heart Disease Detection - Multi Model Demo")

st.write("Enter patient details for prediction:")

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

# --- Prepare input dataframe ---
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

# --- Load all models ---
# For demo, same file renamed; in real scenario, each would be different
model_files = {
    "Logistic Regression": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl",
    "Decision Tree": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl",
    "K-Nearest Neighbor": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl",
    "Naive Bayes": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl",
    "Random Forest": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl",
    "XGBoost": "/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl"
}

# --- Load all models ---
models = {}
for name, file in model_files.items():
    models[name] = joblib.load(file)

# --- Load test data for metrics ---
X_test = pd.read_csv("/mount/src/ml-experiments/01_ClassificationModels/X_test.csv")
y_test = pd.read_csv("/mount/src/ml-experiments/01_ClassificationModels/y_test.csv").values.ravel()

# --- Predict button ---
if st.button("Predict with all models"):
    # Patient prediction
    results = []
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][pred]
        risk = "High Risk" if pred == 1 else "Low Risk"
        results.append({"Model": name, "Prediction": risk, "Probability": f"{prob:.2f}"})
    
    st.write("### Predictions for input patient")
    st.table(pd.DataFrame(results))

    # Model evaluation metrics
    metrics_list = []
    for name, model in models.items():
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
    
    st.write("### Model Evaluation Metrics on Test Set")
    st.table(pd.DataFrame(metrics_list))


