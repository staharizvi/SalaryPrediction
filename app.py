import streamlit as st
import mod
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.title("Salary Prediction ")
st.write('## Welcome to Salary predictor')
st.write('**Note:** This salary predictor is trained on this dataset with multiple fields and requires you to give input in order to predict salary for you.')

# Load the encoder and model
encoder_path = 'encoder.pkl'
model_path = 'model.pkl'

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)


path = 'Salary Prediction of Data Professions.csv'
df = mod.typeCast(mod.readDf(path))
st.dataframe(df.head())

st.write('### Predict your expected salary at our company')
st.write('Enter the required info below & get your prediction')
col1, col2, col3 = st.columns(3)
with col1:
    sex = st.selectbox('Gender',['M','F'])
    designation = st.selectbox('Designation',['Analyst','Senior Analyst','Senior Manager','Director','Associate','Manager'])
    rating = st.number_input('Rating',min_value=0, max_value=5, step=1 )

with col2:
    age = st.number_input('Age', min_value=18, max_value=120, step=1)
    unit = st.selectbox('Unit', ['Marketing', 'Management', 'IT', 'Operations','Web','Finance' ])
    leaveUsed = st.number_input('Leaves Used', min_value=0, max_value=30, step=1)
    

with col3:
    pastexp = st.number_input('Past Exp', min_value=0, max_value=30, step=1)
    service = st.number_input('Service (DAYS)', min_value=0, max_value=2500, step=1)

leaveRemain = 30-leaveUsed
inputData = pd.DataFrame({'SEX': [sex] , 'DESIGNATION': [designation] ,'UNIT':[unit] })

input_encoded = encoder.transform(inputData)
input_encoded_df = pd.DataFrame(input_encoded)
input_final = pd.concat([pd.DataFrame({'AGE':[age] ,'LEAVES USED':[leaveUsed],
                           'LEAVES REMAINING':[leaveRemain] , 'RATINGS':[rating] , 
                            'PAST EXP':[pastexp], 'SERVICE':[service]}),input_encoded_df], axis=1)
input_final.columns = input_final.columns.astype(str)

if st.button('Predict'):
    prediction = model.predict(input_final)
    st.success(f'**Predicted Salary: {prediction[0]}**')