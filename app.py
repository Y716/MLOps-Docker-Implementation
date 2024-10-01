# app.py

from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.joblib"

def train_model():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize and train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    # Load the trained model from the file
    return joblib.load(MODEL_PATH)

# Load the model at startup
model = load_model()

@app.route('/')
def home():
    return "Iris Model API. Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Expecting a JSON object with "features": [list of four features]
        features = data['features']
        if len(features) != 4:
            return jsonify({'error': 'Four features are required.'}), 400

        # Make prediction
        prediction = model.predict([features])
        iris = load_iris()
        species = iris.target_names[prediction[0]]

        return jsonify({'prediction': species})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
