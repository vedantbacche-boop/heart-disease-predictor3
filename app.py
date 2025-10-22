from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load retrained model
model_path = os.path.join(os.path.dirname(__file__), "model_randomforest.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']

        input_data = [
            float(request.form[f]) if f in ['age','trestbps','chol','thalach','oldpeak']
            else int(request.form[f]) for f in features
        ]

        user_data = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(user_data)[0]
        result = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected."
        return render_template('predict.html', prediction_text=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
