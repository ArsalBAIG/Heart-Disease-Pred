import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# Load Model

rf_model = pickle.load(open('rfc_model.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model (1).pkl', 'rb'))

img = Image.open('Heart_pic.webp')


# Importing the function.

import numpy as np

def predict(rf_classifier,male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
# Encoding Categorical Cols.
    male_encoded = 1 if male.lower() == 'male' else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == 'yes' else 0
    BPMeds_encoded = 1 if BPMeds.lower() == 'yes' else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == 'yes' else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == 'yes' else 0
    diabetes_encoded = 1 if diabetes.lower() == 'yes' else 0
# Making a 2D Array.
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded, prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    scaled_features = scaler_model.transform(features)
    result = rf_model.predict(scaled_features)
    return result

male = 'female'
age = 56.00
currentSmoker = 'yes'
cigsPerDay = 3.00
BPMeds = 'no'
prevalentStroke = 'no'
prevalentHyp = 'yes'
diabetes = 'no'
totChol = 285.00
sysBP = 145.00
diaBP = 100.00
BMI = 30.14
heartRate = 80.00
glucose = 86.00

result = predict(rf_model, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
print(result)


if result == 1:
    print("The person has heart disease.")
else:
    print("The person does not have heart disease.")


# Creating Sidebar..

with st.sidebar:
    st.write('Enter Details: ')
    male = st.selectbox('Enter Gender:', ['Male', 'Female'])
    age = st.selectbox('Enter Age: ', ['18_25', '26_35', '36_45', '46_55', '56_65', '66_75', '75 above'])
    currentSmoker = st.selectbox('Did you smoke:', ['Yes', 'No'])
    cigsPerDay = st.selectbox('How many cigs per day:', ['0', '5', '8', '13', '15 above'])
    BPMeds = st.selectbox('Are you on BPMeds:', ['Yes', 'No'])
    prevalentStroke = st.selectbox('Did you face a stroke:', ['Yes', 'No'])
    prevalentHyp = st.selectbox('Did you have Hypertension:', ['Yes', 'No'])
    diabetes = st.selectbox('Have Diabetes:', ['Yes', 'No'])
    totChol = st.selectbox('Total Cholesterol:', ['1000_1500', '1500_2500', '2500_3000'])
    sysBP = st.selectbox('Systolic BP:', ['150_200', '200_250', '250_300', '300 above'])
    diaBP = st.selectbox('Diastolic BP:', ['150_200', '200_250', '250_300', '300 above'])
    BMI = st.selectbox('Body Mass Index:', ['55_65', '66_75', '76_85', '100 above'])
    heartRate = st.selectbox('HeartBeat Rate:', ['91', '102', '112', '132', '150 above'])
    glucose = st.selectbox('Glucose Level:', ['125_135', '135_145', '146_155', '155 above'])


# Keyword Arguments:

predicted_class = predict(rf_classifier= 'rf_model',
                            male= 'Female',
                            age= '46_55',
                            currentSmoker= 'No',
                            cigsPerDay= '0',
                            BPMeds= 'Yes',
                            prevalentStroke= 'No',
                            prevalentHyp= 'Yes',
                            diabetes= 'Yes',
                            totChol= '2500',
                            sysBP= '200_250',
                            diaBP= '150_200',
                            BMI= '76_85',
                            heartRate= '112',
                            glucose= '146_155')


if predicted_class[0]  == 0:
    print('The patient has Heart Problem')
else:
    print('The doesnt have any Heart Problem')

# Creating Web App:

st.title('Heart Disease Prediction Model')
st.image(img)

if st.button('Predict'):
    predicted_class = predict(
        rf_classifier= rf_model,
        male= male,
        age= age,
        currentSmoker= currentSmoker,
        cigsPerDay= cigsPerDay,
        BPMeds= BPMeds,
        prevalentStroke= prevalentStroke,
        prevalentHyp= prevalentHyp,
        diabetes= diabetes,
        totChol= totChol,
        sysBP= sysBP,
        diaBP= diaBP,
        BMI= BMI,
        heartRate= heartRate,
        glucose= glucose
    )

    
    if predicted_class[0]  == 0:
        st.error('The patient has Heart Problem')
    else:
        st.success('The doesnt have any Heart Problem')