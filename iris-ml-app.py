#pip install streamlit

import gc
import os
import re
import time
import math
import joblib
import hyperopt
import pandas as pd 
import numpy as np

import scipy.stats as st
from plotly import tools
import seaborn as sns
from sklearn.utils import resample
from datetime import datetime 
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
from warnings import simplefilter
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import ( FeatureUnion, Pipeline )
from scipy.stats import ( beta, expon, randint, uniform)
from sklearn.base import (TransformerMixin, BaseEstimator)
from plotly.offline import ( download_plotlyjs, init_notebook_mode, plot, iplot )
init_notebook_mode(connected=True)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ( roc_curve, auc, accuracy_score, roc_auc_score,log_loss,confusion_matrix,classification_report)
from sklearn.ensemble import ( RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.preprocessing import ( LabelEncoder, StandardScaler )
from sklearn.model_selection import ( GridSearchCV, StratifiedKFold, train_test_split, cross_val_score, RandomizedSearchCV, KFold )
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
