from flask import Flask, render_template, request
import pickle
import pandas as pd
import os  # <--- needed to get Render's port

app = Flask(__name__)

# Load the retrained 13-feature model
with open("model_randomforest.pkl", "rb") as f:
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
        # Get input values
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
        featureX = float(request.form['featureX'])  # replace with actual name
        featureY = float(request.form['featureY'])  # replace with actual name

        user_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, featureX, featureY]],
                                 columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                          'thalach','exang','oldpeak','slope','featureX','featureY'])

        prediction = model.predict(user_data)[0]
        result = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected."

        return render_template('predict.html', prediction_text=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Get Render's assigned port or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    # Run app with host 0.0.0.0 so Render can access it
    app.run(host="0.0.0.0", port=port)
