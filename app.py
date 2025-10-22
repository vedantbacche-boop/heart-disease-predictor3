from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model_path = "model_randomforest.pkl"   # <-- updated filename
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form inputs
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])

            # Prepare DataFrame
            input_df = pd.DataFrame([[
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]], columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                         'thalach','exang','oldpeak','slope','ca','thal'])

            pred = model.predict(input_df)[0]
            prediction = "Heart disease detected" if pred == 1 else "No heart disease"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

# Run Flask app on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
