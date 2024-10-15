import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and encoders
model = load_model('my_model.h5')
one_hot_encoder = joblib.load('one_hot_encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Define preprocess_data function
def preprocess_data(data):
    # Apply one-hot encoding to 'Admin1Name'
    admin1_encoded = one_hot_encoder.transform(data[['Admin1Name']])

    # Normalize the numerical features
    numerical_features = data[['Weeks', 'Year', 'Month', 'Fatalities']].values
    numerical_normalized = scaler.transform(numerical_features)

    # Combine the one-hot encoded and normalized features
    processed_data = np.concatenate([admin1_encoded, numerical_normalized], axis=1)
    return processed_data

# Streamlit UI
st.title('Prediction of Dengue Outbreak using ANN')

# Input fields
admin1_names = one_hot_encoder.categories_[0].tolist()  # Extract the categories from the encoder
admin1_name = st.selectbox('Select Region', admin1_names)
weeks = st.number_input('Enter Weeks', min_value=1, max_value=52)
year = st.number_input('Enter Year', min_value=2000, max_value=2023)  # Adjust according to your dataset
month = st.number_input('Enter Month', min_value=1, max_value=12)
fatalities = st.number_input('Enter Fatalities', min_value=0, max_value=1000)

submit = st.button('Predict Dengue Outbreak')

# Prediction
if submit:
    # Data preprocessing
    input_data = pd.DataFrame([[admin1_name, weeks, year, month, fatalities]], 
                              columns=['Admin1Name', 'Weeks', 'Year', 'Month', 'Fatalities'])
    processed_data = preprocess_data(input_data)

    # Making prediction
    prediction = model.predict(processed_data)
    st.write(f'Predicted Dengue Outbreak Count: {prediction[0][0]}')  # Access the first element in the prediction array

# Save this script as app.py and run it using Streamlit
