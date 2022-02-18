import streamlit as st
import pandas as pd
import numpy as np

# @st.cache

st.title('Hackaton III ')

df = pd.read_csv("patient.csv")

with st.form("my_form"):
    st.write("Choose a patient:")
    patient_choose = st.selectbox("Choose a patient", df)
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write(f'your choose:  {patient_choose}')
if submit_button:
    execute_button = st.button(f'execute feature extraction for {patient_choose}')
    prediction_button = st.button(f'prediction model for {patient_choose}')
    view_button = st.button(f'view patient scans for {patient_choose}')
