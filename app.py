import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

# App Title
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's details to predict the risk of diabetes.")

# User Input Fields
glucose = st.number_input("Glucose Level (e.g., 70-200)", min_value=0, value=100)
bmi = st.number_input("BMI (Body Mass Index, e.g., 18-50)", min_value=0.0, value=25.0)
age = st.number_input("Age (Years)", min_value=0, value=30)

if st.button("Predict"):
    # We use average values for the other 5 features to help the model predict accurately
    # Features: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
    # Default values used: Pregnancies=3, BP=72, Skin=23, Insulin=80, DPF=0.47
    features = np.array([[3, glucose, 72, 23, 80, bmi, 0.47, age]])
    
    # Scale the input
    scaled_data = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("### Result: High Risk of Diabetes")
        st.write("The model suggests a high probability of diabetes. Please consult a doctor.")
    else:
        st.success("### Result: Low Risk of Diabetes")
        st.write("The model suggests a low probability of diabetes. Maintain a healthy lifestyle!")
