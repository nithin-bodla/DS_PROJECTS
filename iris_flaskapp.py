import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/Iris.csv")
# print(df.head())
df.drop('Id', axis=1, inplace =True)

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
print(model.fit(X_train, y_train))

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model accuracy: {accuracy *100:.2f}%")

import pickle

with open("classifier1.pkl","wb") as model_file:
    pickle.dump(model, model_file)

app = Flask(__name__)

#Load model
with open("classifier1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/predict_api' ,methods=['POST'])
def predict_api():
    try:
        # Call the mthod correctly with()
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        return jsonify({'predicted_species': prediction})
    except Exception as e:
        return jsonify({'error' : str(e)})

if __name__ == '__main__':
    app.run(debug=True)
