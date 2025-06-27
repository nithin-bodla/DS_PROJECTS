# Train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Create simple data
data = pd.DataFrame({
    'number' : list(range(1,101)), 
    'label' : [1 if i%2==0 else 0 for i in range(1,101)]
})

print(data.head())


## dividing our data into independent and dependent features

X = data[['number']]
y = data['label']

# Train a simple model
model = RandomForestClassifier()
print(model.fit(X,y))

# Save the model
with open("even_odd_model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model trained and saved as even_odd_model.pkl")


# even_odd_streamlit.py
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("even_odd_model.pkl","rb") as f:
    model = pickle.load(f)

st.title("Even or Odd Number Predictor")
st.write("Enter a number, and the model will predict if it's even or odd.")

number = st.number_input("Enter a number:" , min_value=0,step=1)

if st.button("Predict"):
    prediction = model.predict(np.array([[number]]))[0]
    print(prediction)
    result = "Even" if prediction == 1 else "Odd"
    st.success(f"The number {number} is **{result}**.")