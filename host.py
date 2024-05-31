import streamlit as st
import joblib
import pandas as pd
model = joblib.load('model.pkl')

jobs = ['blue-collar', 'services', 'admin.', 'entrepreneur',
       'self-employed', 'technician', 'management', 'student', 'retired',
       'housemaid', 'unemployed', 'unknown']
marital_status = ['married', 'single', 'divorced', 'unknown']
education = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
default = ['no', 'unknown', 'yes']
housing = ['yes', 'no', 'unknown']
loan = ['no', 'unknown', 'yes']
contact = ['cellular', 'telephone']
months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
poutcome = ['nonexistent', 'failure', 'success']
st.title("Term Deposit Subscription Guesser")

getInput = False
st.subheader('Enter Data Manually')

column1, column2, column3 = st.columns(3)

with column1:
    age = st.number_input('Age', min_value=0)
    job = st.selectbox('Job', jobs)
    marital = st.selectbox('Marital Status', marital_status)
    education = st.selectbox('Education level', education)
    default = st.selectbox('Credit in Default', default)
    housing = st.selectbox('Housing Loan', housing)
    loan = st.selectbox('Personal Loan', loan)
    contact = st.selectbox('Contact', contact)
with column2:
    month = st.selectbox('Month', months)
    day_of_week = st.selectbox('Day of Week', days)
    duration = st.number_input('Duration of Last Contact', min_value=0)
    campaign = st.number_input('Contacts During Campaign', min_value=0)
    pdays = st.number_input('#Days Passed After Last Contact', min_value=0)
    previous = st.number_input('#Previously Contacted', min_value=0)
    poutcome = st.selectbox('Previous Campaign Outcome', poutcome)
with column3:
    emp_var_rate = st.number_input('Employment Variation Rate')
    cons_price_idx = st.number_input('Consumer Price Index')
    cons_conf_idx = st.number_input('Consumer Confidence Index')
    euribor3m = st.number_input('Euribor 3 Month Rate')
    nr_employed = st.number_input('Number of Employees')

csv = False
if csv:
    st.subheader('Enter CSV Line')
    csv_input = st.text_input('Enter values separated by semicolons:')
    if csv_input:
        try:
            csv_values = [value.strip() for value in csv_input.split(';')]
            if len(csv_values) == 20:
                age = int(csv_values[0])
                job = csv_values[1]
                marital = csv_values[2]
                education = csv_values[3]
                default = csv_values[4]
                housing = csv_values[5]
                loan = csv_values[6]
                contact = csv_values[7]
                month = csv_values[8]
                day_of_week = csv_values[9]
                duration = int(csv_values[10])
                campaign = int(csv_values[11])
                pdays = int(csv_values[12])
                previous = int(csv_values[13])
                poutcome = csv_values[14]
                emp_var_rate = float(csv_values[15])
                cons_price_idx = float(csv_values[16])
                cons_conf_idx = float(csv_values[17])
                euribor3m = float(csv_values[18])
                nr_employed = float(csv_values[19])
            else:
                st.error(f"Please enter exactly 20 values.")
        except ValueError:
            st.error("Please enter valid values separated by semicolons.")

if st.button('Predict'):
    input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed]
    })
    prediction = model.predict(input_data)
    st.subheader('Prediction')
    if(prediction[0] == 1):
        st.write("The model predicts that you subscribed!")
    else:
        st.write("The model predicts that you didn't subscribe.")
