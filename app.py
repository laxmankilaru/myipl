from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any domain

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON
        data = request.get_json()
        print("Received data:", data)

        # Convert to DataFrame for model input
        input_df = pd.DataFrame([data])

        # Predict win probability
        prediction = model.predict_proba(input_df)[0]
        win_percent = round(prediction[1] * 100, 2)
        lose_percent = round(prediction[0] * 100, 2)

        return jsonify({
            "win": win_percent,
            "lose": lose_percent
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
