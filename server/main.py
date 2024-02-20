import streamlit as st
import numpy as np
import models.linear as linear
import models.neural_network as neuro

st.sidebar.header("Enter client details:")

age = st.sidebar.slider('Age', 26, 51)

bmi = st.sidebar.number_input('Body mass index', min_value=26)

sex_values = ['Man', 'Woman']
sex_val = st.sidebar.selectbox('Sex', options=sex_values)

child_val = st.sidebar.toggle('Have children')

button = st.sidebar.button("Calculate the cost of insurance")
col1, col2 = st.columns(2)

col1.header("Linear model")
col1.write("Calculation based on the client's age")
col1.write(f"R² = {linear.get_r2()}")
col1.write(f"RMSE = {linear.get_rmse()}")

col2.header("Neural network")
col2.write("Calculation based on all data")
col2.write(f"R² = {neuro.get_r2()}")
col2.write(f"RMSE = {neuro.get_rmse()}")

if button:
    col1.write("<b>Cost of insurance: $" + str(np.round(linear.calculate(age), 2)) + "</b> <hr>",
               unsafe_allow_html=True)
    neuro_ans, neuro_normX, neuro_normY = neuro.calculate(age, child_val, bmi, sex_val)
    col2.write("<b>Cost of insurance: $" + str(np.round(neuro_ans, 2)) + "</b> <hr>", unsafe_allow_html=True)
    col1.write(
        f"Client's data: <table><tr><th>age</th><td>{age}</td></tr><tr><th>sex</th><td>{neuro.def_sex(sex_val)}</td></tr><tr><th>bmi</th><td>{round(bmi, 2)}</td></tr><tr><th>children</th><td>{neuro.def_children(child_val)}</td></tr></table>",
        unsafe_allow_html=True)
    col1.write(f"Predicted result: ${age * 263 - 3781}")
    col2.write(
        f"Normalized data: <table><tr><th>age</th><td>{neuro_normX['age'][0]}</td></tr><tr><th>sex</th><td>{neuro_normX['sex_male'][0]}</td></tr><tr><th>bmi</th><td>{neuro_normX['bmi'][0]}</td></tr><tr><th>children</th><td>{neuro_normX['children'][0]}</td></tr><tr><th>charges</th><td>{neuro_normY[0][0]}</td></tr></table>",
        unsafe_allow_html=True)
