import streamlit as st
import pickle
import numpy as np

# Load files
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except:
    st.error("Model files not found. Please check filenames.")

st.title("🩺 Diabetes Prediction App")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    # Change the number of zeros here to match your model's columns
    # If your model has 8 columns, you need 8 values
    features = np.array([[0, glucose, 0, 0, 0, bmi, 0, age]]) 
    scaled_data = scaler.transform(features)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
