import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.title('Penguins Specie Prediction')

st.markdown('''
            
            ![image of penguins](images/download.jpg)
Predicting the Species of $Penguins$ Using the penguins dataset


### Fill up the following features
            ''')
# @st.cache
# def load_model():
#     return joblib.load('pipeline.pkl')

feature_names = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                         'body_mass_g', 'sex']

model = joblib.load('pipeline.pkl')

island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
bill_length_mm = st.slider('Bill Length (mm) ', 30, 60)
bill_depth_mm = st.slider('Bill Depth (mm) ', 13, 22)
flipper_length_mm = st.slider('Flipper Length (mm) ', 160, 240)
body_mass_g = st.slider('Body Mass (g) ', 2500, 6500)
sex = st.radio('Sex', ('Male', 'Female'))

predict_button = st.button('Predict')


def predict(data):
    if predict_button:
        
        model_input = pd.Series(data, index=feature_names).to_frame().T
        prediction = model.predict(model_input)

        # print(prediction[0])
        st.markdown(f'**Specie Predicted as** ${prediction[0]}$' )
        

features = [island, bill_length_mm, bill_depth_mm,
            flipper_length_mm, body_mass_g, sex]
# print(features)
predict(features)


# test_row = data.loc[0].drop('species').to_frame().T
# print(model.predict(test_row))  # returns 'Adelie'
