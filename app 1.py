import numpy as np
import pickle
import streamlit as st

# Load the saved model safely
try:
    loaded_model = pickle.load(open(r'C:\Users\SruthiP\OneDrive - theproindia.com\Documents\python\Pro-Friday\trained_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    loaded_model = None

# Function for Prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Ensure float conversion
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    try:
        prediction = loaded_model.predict(input_data_reshaped)
        return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
    except Exception as e:
        return f"Prediction error: {e}"

# Main function
def main():
    st.title('Diabetes Prediction Web App')

    # Get user inputs
    Pregnancies = float(st.text_input('Number of Pregnancies', '0'))
    Glucose = float(st.text_input('Glucose Level', '0'))
    BloodPressure = float(st.text_input('Blood Pressure value', '0'))
    SkinThickness = float(st.text_input('Skin Thickness value', '0'))
    Insulin = float(st.text_input('Insulin Level', '0'))
    BMI = float(st.text_input('BMI value', '0'))
    DiabetesPedigreeFunction = float(st.text_input('Diabetes Pedigree Function value', '0'))
    Age = float(st.text_input('Age of the Person', '0'))

    # Make prediction
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        if loaded_model:
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        else:
            diagnosis = "Model not loaded."

    if diagnosis == 'The person is diabetic':
        st.error(diagnosis)
    else:
        st.success(diagnosis)

if __name__ == '__main__':
    main()