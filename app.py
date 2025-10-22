from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Heart Disease Prediction API is running successfully on Render! 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input
        data = request.get_json()

        # Extract features
        features = [[
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]]

        # Convert to DataFrame
        columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
                   'thalach','exang','oldpeak','slope','ca','thal']
        input_df = pd.DataFrame(features, columns=columns)

        # Predict
        prediction = model.predict(input_df)[0]

        # Return response
        result = "Heart disease detected" if prediction == 1 else "No heart disease"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
