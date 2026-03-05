import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = pickle.load(open("model/diabetes_model.pk1", "rb"))
scaler = pickle.load(open("model/scaler.pk1", "rb"))

# -----------------------------
# Title Section
# -----------------------------

st.title("🩺 Diabetes Risk Prediction Dashboard")
st.write(
"""
This tool predicts **diabetes risk** based on patient health metrics.

Dataset used: Pima Indians Diabetes Dataset  
Models evaluated: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost
"""
)

st.divider()

# -----------------------------
# Input Layout (2 columns)
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 0)
    glucose = st.slider("Glucose Level", 0, 200, 120)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin Level", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 1, 100, 30)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Diabetes Risk"):

    data = np.array([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]])

    data_sclaed = scaler.transform(data)
    prediction = model.predict(data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
        st.progress(80)

    else:
        st.success("Low Risk of Diabetes")
        st.progress(30)

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
"""
This project demonstrates a machine learning workflow including  
data preprocessing, model comparison, and deployment using Streamlit.
"""
)