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
 Peter B. Kaaya
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        preg = st.sidebar.slider('preg', 4.7, 15.9, 8.31)
        plas = st.sidebar.slider('plas', 0.12,1.58 , 0.52)
        pres = st.sidebar.slider('pres', 0.0,1.0 , 0.5)
        skin = st.sidebar.slider('skin', 0.01,0.6 , 0.08)
        insulin=st.sidebar.slider('insulin', 6.0,289.0 , 46.0)
        mass=st.sidebar.slider('mass', 8.4,14.9, 10.4)
        pedi=st.sidebar.slider('pedi', 0.33,2.0,0.65 )
        age=st.sidebar.slider('age', 0.33,2.0,0.65 )
        data = {'preg': preg,
                'plas': plas,
                'pres': pres,
                'skin': skin,
              'insulin':insulin,
              'mass':mass,
                'pedi':pedi
               }
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
#reading csv file
data=pd.read_csv("./fulltrainset.csv")
X =np.array(data[['preg', 'plas' , 'pres' , 'skin' , 'insulin' , 'mass' , 'pedi']])
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
