import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load real-world Diabetes dataset (442 sampled, 10 Features)
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = np.sqrt(mean_squared_error(y_test, y_pred))
st.sidebar.write(f"✅ Model RMSE: {mse:.2f}")

# Save the model
with open("diabetes_model.pkl","wb") as f:
    pickle.dump(model,f)

# Streamlit App Layout
st.title("Diabetes Progression Predictor")

st.markdown(""" 
Enter the following physiological measurements to predict the disease progression measure one year after baseline.
""")

input_data = {}
for feat in X.columns:
    val = st.slider(
        feat,
        float(X[feat].min()),
        float(X[feat].max()),
        float(X[feat].mean())
    )
    input_data[feat]=val

if st.button("Predict"):
    model = pickle.load(open("diabetes_model.pkl","rb"))
    df_input = pd.DataFrame([input_data])
    pred = model.predict(df_input)[0]
    st.subheader(f"🩺 Predicted diabetes progression: {pred:.2f}")
    st.write("Lower is generally better-progression is measured quantitatively.")