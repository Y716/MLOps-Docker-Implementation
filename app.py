# app.py

from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

MODELS_DIR = "models"

def ensure_models_dir():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created models directory at {MODELS_DIR}")

def get_next_version():
    existing_models = [f for f in os.listdir(MODELS_DIR) if f.startswith("model_v") and f.endswith(".joblib")]
    versions = [int(f.split("_v")[1].split(".joblib")[0]) for f in existing_models if f.split("_v")[1].split(".joblib")[0].isdigit()]
    next_version = max(versions) + 1 if versions else 1
    return next_version

def train_model():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize and train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Determine the next version number
    version = get_next_version()
    model_filename = f"model_v{version}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)

    # Save the trained model to the models directory
    joblib.dump(model, model_path)
    print(f"Model trained and saved as {model_filename}.")

def load_latest_model():
    existing_models = [f for f in os.listdir(MODELS_DIR) if f.startswith("model_v") and f.endswith(".joblib")]
    if not existing_models:
        print("No existing models found. Training a new model.")
        train_model()
        existing_models = [f for f in os.listdir(MODELS_DIR) if f.startswith("model_v") and f.endswith(".joblib")]

    # Extract version numbers and find the latest
    models_with_versions = []
    for model_file in existing_models:
        try:
            version = int(model_file.split("_v")[1].split(".joblib")[0])
            models_with_versions.append((version, model_file))
        except (IndexError, ValueError):
            continue  # Skip files that don't match the pattern

    if not models_with_versions:
        raise FileNotFoundError("No valid model files found.")

    # Get the model with the highest version number
    latest_version, latest_model = max(models_with_versions, key=lambda x: x[0])
    latest_model_path = os.path.join(MODELS_DIR, latest_model)
    print(f"Loading the latest model: {latest_model}")
    return joblib.load(latest_model_path)

# Ensure the models directory exists
ensure_models_dir()

# Load the latest model at startup
model = load_latest_model()

@app.route('/')
def home():
    return "Iris Model API. Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Expecting a JSON object with "features": [list of four features]
        features = data.get('features', [])
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
