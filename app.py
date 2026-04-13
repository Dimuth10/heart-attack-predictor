from flask import Flask, render_template, request
from pathlib import Path
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_attack_model.pkl"
COLS_PATH  = BASE_DIR / "model" / "feature_columns.pkl"

model    = joblib.load(MODEL_PATH)
features = joblib.load(COLS_PATH)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            user_input = {
                'age':      float(request.form['age']),
                'sex':      request.form['sex'],
                'cp':       request.form['cp'],
                'trestbps': float(request.form['trestbps']),
                'chol':     float(request.form['chol']),
                'fbs':      request.form['fbs'],
                'restecg':  request.form['restecg'],
                'thalach':  float(request.form['thalach']),
                'exang':    request.form['exang'],
                'oldpeak':  float(request.form['oldpeak']),
                'slope':    request.form['slope'],
                'ca':       float(request.form['ca']),
                'thal':     request.form['thal'],
            }

            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=features, fill_value=0)

            probability     = model.predict_proba(input_df)[0][1]
            probability_pct = round(probability * 100, 2)
            risk_level      = "High Risk" if probability_pct >= 50 else "Low Risk"

            return render_template('result.html',
                                   probability=probability_pct,
                                   risk_level=risk_level)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)