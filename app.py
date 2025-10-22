from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
model_path = "model_randomforest.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Fields for prediction form
fields = {
    'age': 'Age',
    'sex': 'Sex (0=female,1=male)',
    'cp': 'Chest Pain type (0-3)',
    'trestbps': 'Resting BP',
    'chol': 'Cholesterol',
    'fbs': 'Fasting blood sugar > 120',
    'restecg': 'Resting ECG (0-2)',
    'thalach': 'Max heart rate achieved',
    'exang': 'Exercise induced angina (0=no,1=yes)',
    'oldpeak': 'ST depression',
    'slope': 'Slope (0-2)',
    'ca': 'Major vessels colored (0-4)',
    'thal': 'Thal (0=normal,1=fixed,2=reversable)'
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    try:
        if request.method == "POST" and model:
            input_data = [float(request.form[field]) for field in fields]
            df = pd.DataFrame([input_data], columns=fields.keys())
            pred = model.predict(df)[0]
            prediction = "Heart disease detected" if pred == 1 else "No heart disease"

        return render_template("predict.html", fields=fields, prediction=prediction, model=model)

    except Exception as e:
        return f"<h3>Something went wrong:</h3><p>{e}</p>"

@app.route("/awareness")
def awareness():
    return render_template("awareness.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
