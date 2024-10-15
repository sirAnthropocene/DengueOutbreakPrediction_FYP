import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the OneHotEncoder and MinMaxScaler
one_hot_encoder = joblib.load('one_hot_encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Load the trained ANN model
model = load_model('my_model.h5')

def preprocess_data(data):
    # Convert 'PeriodStartDate' to datetime
    data['PeriodStartDate'] = pd.to_datetime(data['PeriodStartDate'], format='%d/%m/%Y')
    
    # Extract Year and Month
    data['Year'] = data['PeriodStartDate'].dt.year
    data['Month'] = data['PeriodStartDate'].dt.month
    
    # Drop irrelevant columns
    data_cleaned = data.drop(['ConditionName', 'ConditionSNOMED', 'PathogenName', 'PathogenTaxonID', 
                              'CountryName', 'CountryISO', 'PeriodEndDate'], axis=1)
    
    # One-Hot Encode 'Admin1Name'
    admin1_encoded = one_hot_encoder.transform(data_cleaned[['Admin1Name']])
    admin1_encoded_df = pd.DataFrame(admin1_encoded, columns=one_hot_encoder.get_feature_names_out(['Admin1Name']))
    
    # Normalize numerical features
    numerical_features = ['Weeks', 'Year', 'Month', 'Fatalities']
    data_cleaned[numerical_features] = scaler.transform(data_cleaned[numerical_features])
    
    # Combine one-hot encoded and normalized features
    processed_data = pd.concat([data_cleaned.drop(['Admin1Name'], axis=1), admin1_encoded_df], axis=1)
    
    return processed_data

def predict_dengue_outbreak(data):
    # Preprocess input data
    processed_data = preprocess_data(data)
    
    # Separate features and target variable
    X = processed_data.drop(['CountValue', 'PeriodStartDate'], axis=1)
    
    # Make predictions using the trained model
    predictions = model.predict(X)
    
    return predictions

# Streamlit App
st.title("Dengue Outbreak Prediction System")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Dengue dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read data from uploaded file
    dengue_data = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(dengue_data.head())

    # Button to trigger predictions
    if st.button("Predict Dengue Outbreak"):
        # Make predictions
        predictions = predict_dengue_outbreak(dengue_data)

        # Display predicted values
        st.subheader("Predicted Dengue Outbreak")
        st.write(predictions)

# Note: You can further enhance the app by adding more features, visualizations, and improving the user interface.
