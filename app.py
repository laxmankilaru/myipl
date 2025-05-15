from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load trained pipeline
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to IPL Win Predictor API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({'win_probability': float(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(__import__("os").environ.get('PORT', 5000)))