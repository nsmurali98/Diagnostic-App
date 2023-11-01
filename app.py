import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Medical Diagnostic Web App ⚕️")
st.subheader("Does the patient have Diabetics?")
df = pd.read_csv(r"I:\1 murali\great learning\8 ML Deployment\Day 1\mlop_blr\diabetes.csv")
if st.sidebar.checkbox("View Data",False):
    st.write(df)
if st.sidebar.checkbox("View Distribution",False):
    df.hist()
    plt.tight_layout()
    st.pyplot()

# Step 1: Load the Pickled model
model = open("rfc.pickle","rb")
clf = pickle.load(model)
model.close()

# Step 2: Get the front end user input
pregs = st.number_input('Pregnancies',0,20,0) 
plas = st.slider('Glucose',40,200,40)
pres = st.slider('Blood Pressure',20,150,20) 
skin = st.slider('Skin Thickness',7,99,7) 
insulin = st.slider('Insulin',14,850,14)
bmi = st.slider('BMI',18,70,18) 
dpf = st.slider('Diabetes Pedigree Function',0.05,2.50,0.05) 
age = st.slider('Age',21,90,21)

# Step 3: Get the model input
input_data = [[pregs,plas,pres,skin,insulin,bmi,dpf,age]]

# Step 4: Get the prediction and print the result
prediction = clf.predict(input_data)[0]
if st.button("Predict"):
    if prediction == 0:
        st.subheader("Non Diabetic")
    else:
        st.subheader("Diabetic")
