# import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load wine dataset from sklearn
from sklearn.datasets import load_wine
wine = load_wine()
#print(wine)
df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['target'] = wine.target

# print(df.head())

# Features and labels
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train,y_train)

#Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model accuracy : {accuracy * 100 : .2f}%")

# Save Model
with open("wine_classifier.pkl", "wb") as f:
    pickle.dump(model,f)

# StreamLit UI
import streamlit as st
import numpy as np

# Load model
with open("wine_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Wine Quality Classifier")
st.write("Enter the chemical properties to predict the wine type.")

# User inputs
input_features = []
feature_names = wine.feature_names[:13]

for name in feature_names:
    value = st.slider(f"{name} ", float(df[name].min()), float(df[name].max()))
    input_features.append(value)

# Prediction
if st.button("Predict Wine Type"):
    input_array = np.array([input_features])
    prediction = model.predict(input_array)
    wine_class = wine.target_names[prediction[0]]
    st.write(f" Predicted Wine Class **{wine_class}**")