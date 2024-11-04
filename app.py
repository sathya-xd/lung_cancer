import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = pickle.load(open('C:/Users/SATHYA XD/Desktop/internship/dotch_internship/lung_cancer_prediction_model.pkl', 'rb'))

# Streamlit app
st.title('Lung Cancer Prediction')

# Input fields for the features
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
age = st.slider('Age', 1, 100, 30)
smoking = st.selectbox('Smoking', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
yellow_fingers = st.selectbox('Yellow Fingers', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
anxiety = st.selectbox('Anxiety', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
peer_pressure = st.selectbox('Peer Pressure', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
chronic_disease = st.selectbox('Chronic Disease', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
fatigue = st.selectbox('Fatigue', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
allergy = st.selectbox('Allergy', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
wheezing = st.selectbox('Wheezing', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
alcohol_consuming = st.selectbox('Alcohol Consuming', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
coughing = st.selectbox('Coughing', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
shortness_of_breath = st.selectbox('Shortness of Breath', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
swallowing_difficulty = st.selectbox('Swallowing Difficulty', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
chest_pain = st.selectbox('Chest Pain', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Button to make the prediction
if st.button('Predict'):
    # Collect the input data
    data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    
    # Make the prediction
    prediction = model.predict(data)[0]
    
    # Display the result
    if prediction == 0:
        st.success('The model predicts that you do not have lung cancer.')
    else:
        st.warning('The model predicts that you have lung cancer.')
