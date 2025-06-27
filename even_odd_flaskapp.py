# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

#Load model
with open("even_odd_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        number = int(data['number'])
        prediction = model.predict(np.array([[number]]))[0]
        result = 'Even' if prediction == 1 else 'Odd'
        return jsonify({'number' : number, 'prediction' : result})
    except Exception as e:
        return jsonify({'error' : str(e)})

if __name__ == '__main__':
    app.run(debug=True)