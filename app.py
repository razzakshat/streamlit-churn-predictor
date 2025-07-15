import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle



model = tf.keras.models.load_model('model.h5')

with open('labelencoder.pkl', 'rb') as file:
    LabelEncoder = pickle.load(file)


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)


st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography", options=onehot_encoder.categories_[0])
gender = st.selectbox("Select Gender", LabelEncoder.classes_)
age = st.number_input("Enter Age", min_value=0)
tenure = st.number_input("Enter Tenure (in months)", min_value=0)
balance = st.number_input("Enter Balance", min_value=0.0)
num_of_products = st.number_input("Enter Number of Products", min_value=1)
has_cr_card = st.selectbox("Has Credit Card?", options=[0, 1])
is_active_member = st.selectbox("Is Active Member?", options=[0, 1])
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0)
creditscore = st.number_input('Credit Score')

input_data = { 
    'CreditScore': [creditscore],
    'Gender': LabelEncoder.transform([gender])[0],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]}

input_df = pd.DataFrame(input_data)

geo_encoded = onehot_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))


final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)


input_data_scaled = scaler.transform(final_input)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

