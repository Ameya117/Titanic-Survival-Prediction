import streamlit as st
import pandas as pd
import joblib

model = joblib.load('titanicsurvival.pkl')

st.title('Titanic Survival Prediction')
st.write('Enter the details to predict if the passenger would survive.')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 1000.0, 50.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

input_data['Embarked'] = input_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success('The passenger would survive.')
    else:
        st.error('The passenger would not survive.')