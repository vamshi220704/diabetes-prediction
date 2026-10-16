
import os
import joblib
import streamlit as st
import numpy as np

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Website title
st.title("Diabetes Prediction System")
st.write("Enter the details below to get a model prediction.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=79.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict
if st.button("Predict"):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.warning("Model prediction: Higher likelihood of diabetes.")
        else:
            st.success("Model prediction: Lower likelihood of diabetes.")

        st.caption(
            "This is an educational machine-learning prediction, "
            "not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
