#preg	plas	pres	skin	insulin	mass	pedi	age	result
#import libraries
#conda install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#app heading
st.write("""
# An Ensemble Machine Learning Based model for predicting predisposition to diabetic Condition
""")
st.write("""
 *Peter B. Kaaya*
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        preg = st.sidebar.slider('Pregnancies', 0, 10, 20)
        plas = st.sidebar.slider('Glucose', 0,100 , 200)
        pres = st.sidebar.slider('BloodPressure', 0.0,55.0 , 110.0)
        skin = st.sidebar.slider('SkinThickness', 0.0,35.0 , 70.0)
        insulin=st.sidebar.slider('Insulin', 0.0,450.0 , 900.0)
        mass=st.sidebar.slider('BMI', 0.0,30.5, 61.5)
        pedi=st.sidebar.slider('DiabetesPedigreeFunction', 0.0,1.55,3.55 )
        age=st.sidebar.slider('Age', 0,60,120 )
        data = {'preg': preg,
                'plas': plas,
                'pres': pres,
                'skin': skin,
              'insulin':insulin,
              'mass':mass,
                'pedi':pedi,
                'age':age
               }
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
#reading csv file
data=pd.read_csv("./fulltrainset.csv")
X =np.array(data[['preg', 'plas' , 'pres' , 'skin' , 'insulin' , 'mass' , 'pedi', 'age']])
Y = np.array(data['result'])
#random forest model
rfc= RandomForestClassifier()
rfc.fit(X, Y)
st.subheader('Diabetic Condition their corresponding result, 1 "POSITIVE", 0 "NEGATIVE"')
st.write(pd.DataFrame({
   'Result': [0,1]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
