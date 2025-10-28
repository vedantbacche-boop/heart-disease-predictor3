from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ✅ Load the model safely
model_path = os.path.join(os.path.dirname(__file__), "model_randomforest.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ✅ Home (Predict) Page
@app.route('/')
def home():
    return render_template('predict.html', title="Heart Disease Prediction")

# ✅ About Page
@app.route('/about')
def about():
    return render_template('about.html', title="About HeartCare AI")

# ✅ Awareness Page
@app.route('/awareness')
def awareness():
    return render_template('awareness.html', title="Heart Health Awareness")

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal'
        ]

        # Extract user inputs from form
        input_data = []
        for f in features:
            val = request.form.get(f)
            if val is None or val == "":
                return render_template('predict.html', prediction_text="⚠️ Please fill all fields.", title="Heart Disease Prediction")
            if f in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                input_data.append(float(val))
            else:
                input_data.append(int(val))

        # Convert to DataFrame
        user_data = pd.DataFrame([input_data], columns=features)

        # Make prediction
        prediction = model.predict(user_data)[0]
        result = "❤️ Heart Disease Detected!" if prediction == 1 else "✅ No Heart Disease Detected."

        return render_template('predict.html', prediction_text=result, title="Heart Disease Prediction")

    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {str(e)}", title="Heart Disease Prediction")

# ✅ Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000
    app.run(host="0.0.0.0", port=port, debug=True)
