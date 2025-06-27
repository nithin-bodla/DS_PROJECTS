from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_wine

app = Flask(__name__)

# Load the model
with open("wine_classifier.pkl", "rb") as f:
    model = pickle.load(f)

wine = load_wine()
feature_names = wine.feature_names
target_names = wine.target_names

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        features = [float(data[feature]) for feature in feature_names]

        input_array = np.array([features])
        prediction = model.predict(input_array)

        return jsonify({
            'predicted_class' : int(prediction[0]),
            'predicted_class_name ': wine.target_names[prediction[0]],
            'input_features' : {feature: data[feature] for feature in feature_names}

        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
