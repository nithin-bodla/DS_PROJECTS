import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("/Users/nithinbodla/Downloads/DataScience/AimNxt_Project/Stremlitapp/data/Iris.csv")

# print(df.head())
df.drop('Id', axis=1, inplace =True)

X = df.drop('Species', axis = 1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
print(model.fit(X_train, y_train))

y_pred = model.predict(X_test)
accurracy = accuracy_score(y_test, y_pred)
print(f"MOdel accuracy: {accurracy *100:.2f}%")

import pickle

with open("classifier.pkl" , "wb") as model_file:
    pickle.dump(model, model_file)

import streamlit as st
import pickle
import numpy as np

with open("classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Iris Species Classifier")
st.write("Enter the Flower measurements to classify the species.")

sepal_length = st.slider("Sepal Length (cm) ", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm) ", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.slider("Petal Length (cm) ", min_value=1.0, max_value=7.0,step=0.1)
petal_width = st.slider("Petal Width (cm) ", min_value=0.1, max_value=2.5, step=0.1)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.write(f"Predicted Iris Species: {prediction[0]}")