from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your saved Random Forest model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Collect all input values from the form
            data = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"]),
            ]

            # Create DataFrame with same column names
            columns = [
                "age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak","slope","ca","thal"
            ]
            user_df = pd.DataFrame([data], columns=columns)

            # Predict
            prediction = model.predict(user_df)[0]
            result = (
                "⚠️ The model predicts that this person may have heart disease."
                if prediction == 1
                else "✅ The model predicts that this person does NOT have heart disease."
            )
        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

    from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


