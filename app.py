from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load the trained model and scaler
model = pickle.load(open('model/model_randomforest.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])

        # Create input array
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope]])

        # Scale input
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        result = model.predict(user_input_scaled)[0]
        prediction_text = "⚠️ Person Has Heart Disease" if result == 1 else "🟢 Person Does Not Have Heart Disease"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# ✅ This part is required for Render to bind the correct host and port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
