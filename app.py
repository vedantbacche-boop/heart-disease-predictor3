from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the retrained 13-feature model
with open("model_randomforest.pkl", "rb") as f:
    model = pickle.load(f)

# Home page → Predict form
@app.route('/')
def home():
    return render_template('predict.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Awareness page
@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

# Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        featureX = float(request.form['featureX'])   # replace with actual name
        featureY = float(request.form['featureY'])   # replace with actual name

        # Create DataFrame
        user_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, featureX, featureY]],
                                 columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                          'thalach','exang','oldpeak','slope','featureX','featureY'])

        # Predict
        prediction = model.predict(user_data)[0]
        result = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected."

        return render_template('predict.html', prediction_text=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
