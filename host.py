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

age = st.number_input('Age', min_value=0)
job = st.selectbox('Job', jobs)
marital = st.selectbox('Marital Status', marital_status)
education = st.selectbox('Education level', education)
default = st.selectbox('Credit in Default', default)
housing = st.selectbox('Housing Loan', housing)
loan = st.selectbox('Personal Loan', loan)
contact = st.selectbox('Contact', contact)
month = st.selectbox('Month', months)
day_of_week = st.selectbox('Day of Week', days)
duration = st.number_input('Duration of Last Contact', min_value=0)
campaign = st.number_input('Contacts During Campaign', min_value=0)
pdays = st.number_input('#Days Passed After Last Contact', min_value=0)
previous = st.number_input('#Previously Contacted', min_value=0)
poutcome = st.selectbox('Previous Campaign Outcome', poutcome)
emp_var_rate = st.number_input('Employment Variation Rate')
cons_price_idx = st.number_input('Consumer Price Index')
cons_conf_idx = st.number_input('Consumer Confidence Index')
euribor3m = st.number_input('Euribor 3 Month Rate')
nr_employed = st.number_input('Number of Employees')

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
