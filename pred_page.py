import streamlit as st
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer as ct
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

flip_path = "C:/Users/saura/Desktop/Flipkart_Laptop_Data Analysis/resources/data/flip.pkl"
rf_path = "C:/Users/saura/Desktop/Flipkart_Laptop_Data Analysis/resources/data/rf.pkl"
model_path = "C:/Users/saura/Desktop/Flipkart_Laptop_Data Analysis/resources/data/model.pkl"


st.set_page_config(page_title="Laptop Predictor",
                   page_icon="💻",
                   layout="wide"
)
st.markdown("""
    <style>
        body {
            background-color: #white;
        }
        .stButton button {
            background-color: #007bff;
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Laptop Price Prediction")

model = pickle.load(open(model_path, 'rb'))

lap = pickle.load(open(flip_path, 'rb'))
rf = pickle.load(open(rf_path, 'rb'))

flip = pd.DataFrame(lap)



features = ["brand", "processor", "ram", "os", "Storage"]
f = flip[["brand", "processor", "ram", "os", "Storage"]]
y = np.log(flip['MRP'])
X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=47)
step1 = ct(transformers=[
    ('encoder',OneHotEncoder(sparse=False,drop='first'),[0,1,2,3,4])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)


brand = st.selectbox("Brand:- ", flip["brand"].unique())
processor = st.selectbox("Processor:- ", flip["processor"].unique())
ram = st.selectbox("RAM:- ", flip["ram"].unique())
os = st.selectbox("Operating Syatem:- ", flip["os"].unique())
Storage = st.selectbox("Storage:- ", flip["Storage"].unique())
butt = st.button("--Predict--")

query = np.array([brand, processor, ram, os, Storage])
query = query.reshape(1, -1)
p = pipe.predict(query)[0]
result = np.exp(p)
st.subheader("Your Predicted Laptop Prize is: ")
st.subheader("₹{}".format(result.round(2)))

st.subheader("Thank You For Visiting!!!")