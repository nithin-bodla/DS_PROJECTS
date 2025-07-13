from flask import Flask, request

import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare the diabetes dataset
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target
columns = X.columns

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model for prediction
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = [data[col] for col in columns]
        input_df = pd.DataFrame([input_values], columns=columns)
        prediction = model.predict(input_df)[0]
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        return {"error": str(e)}, 400
        
if __name__ == '__main__':
    app.run(debug=True)