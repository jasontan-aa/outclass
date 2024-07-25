
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

# Load your H5 model
model = tf.keras.models.load_model('advertising_lr.h5')  # Update with your actual model path

st.write("# Sales Prediction App")
st.write("This app predicts the sales based on TV, Radio, and Newspaper Count")

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    tv = st.sidebar.slider('TV Advertising Budget', 0.0, 300.0, 150.0)  # Change slider range as needed
    radio = st.sidebar.slider('Radio Advertising Budget', 0.0, 50.0, 25.0)  # Change slider range as needed
    newspaper = st.sidebar.slider('Newspaper Advertising Budget', 0.0, 50.0, 25.0)  # Change slider range as needed
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Input Parameters')
st.write(df)

# Prepare the input for prediction
X_new = df.values  # Convert the DataFrame to a NumPy array

# Make prediction using the loaded model
prediction = model.predict(X_new)

st.subheader('Prediction')
st.write(f"Predicted Sales: {prediction[0][0]:.2f}")
