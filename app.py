from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRICE_MODEL_PATH = os.path.join(BASE_DIR, "price_model.joblib")
BRAND_MODEL_PATH = os.path.join(BASE_DIR, "brand_model.joblib")

price_model = joblib.load(PRICE_MODEL_PATH)
brand_model = joblib.load(BRAND_MODEL_PATH)

print("✅ Models loaded successfully")

# -----------------------------------
# ROUTES
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")   # landing page

@app.route("/predict")
def predict_page():
    return render_template("predict.html") # prediction form page

# -----------------------------------
# PRICE PREDICTION API
# -----------------------------------
@app.route("/predict_price", methods=["POST"])
def predict_price():
    data = request.json

    df = pd.DataFrame([data])  # convert JSON to DataFrame

    prediction = price_model.predict(df)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })

# -----------------------------------
# BRAND PREDICTION API
# -----------------------------------
@app.route("/predict_brand", methods=["POST"])
def predict_brand():
    data = request.json

    # Remove 'make' if user sends it
    data.pop("make", None)

    df = pd.DataFrame([data])

    prediction = brand_model.predict(df)[0]

    return jsonify({
        "predicted_brand": prediction
    })

# -----------------------------------
# RUN LOCAL SERVER
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
