
# import the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler


# Load the trained machine learning model
@st.cache_data
def load_model():
    pickle_path = os.path.join(os.getcwd(), 'models','xgb_model.pkl')
    with open(pickle_path, 'rb') as f:
        the_model = pickle.load(f)
    return the_model
model = load_model()

# Load the trained scaling
@st.cache_data
def load_scaler():
    scaler_path = os.path.join(os.getcwd(), 'models','scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler
scaler = load_scaler()

# Define the Streamlit app layout
st.title('Combined Cycle Power Plant Power Prediction')
st.markdown('Welcome to the Combined Cycle Power Plant Power Prediction Web App! '
            'This app utilizes the XGBoost algorithm to predict the electric power output of a combined cycle power plant.')

st.markdown('## About Combined Cycle Power Plants')
st.markdown('Combined cycle power plants are a type of power generation facility that produce electricity through a combination of gas and steam turbines. '
            'They are known for their high efficiency and low emissions, making them an important component of the energy landscape.')

st.markdown('## Power Prediction with XGBRegressor')
st.markdown('XGBRegressor is a powerful machine learning algorithm widely used for regression tasks. '
            'It excels in capturing complex relationships between input features and target variables, making it suitable for predicting power output in this context.')

st.markdown('## Usage')
st.markdown('To use this app, follow these simple steps:')

st.markdown('1. Enter the input feature values:')
feature1 = st.number_input('Temperature (°C)', min_value=0.0)
feature2 = st.number_input('Exhaust Vacuum (cm Hg)', min_value=0.0)
feature3 = st.number_input('Ambient Pressure (millibar)', min_value=0.0)
feature4 = st.number_input('Relative Humidity (%)', min_value=0.0)

st.markdown('2. Click on the "Predict Power" button to initiate the power prediction process.')

# Create a button to trigger the prediction
if st.button('Predict Power'):
    
    input_features = np.array([[feature1, feature2, feature3, feature4]])
    
    # Scale the input features  
    scaled_features = scaler.transform(input_features)  
    
    # Make the prediction using the loaded model
    prediction = model.predict(scaled_features)

    # Display the predicted power value to the user
    st.success(f'Predicted Electric Power Output: {prediction[0]:.5f} MW')


st.markdown('## Get Started')
st.markdown('To get started, make sure you have the necessary input feature values ready. '
            'Then, simply input the values, and let the app perform the power prediction for you.')

