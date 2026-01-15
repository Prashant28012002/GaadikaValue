from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib

app = Flask(__name__)

# -----------------------------------
# LOAD MODELS (RELATIVE PATHS)
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

price_model = joblib.load(os.path.join(BASE_DIR, "price_model.joblib"))
brand_model = joblib.load(os.path.join(BASE_DIR, "brand_model.joblib"))

print("✅ Models loaded successfully")

# -----------------------------------
# ROUTES
# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# -----------------------------------
# PRICE PREDICTION API
# -----------------------------------
@app.route("/predict_price", methods=["POST"])
def predict_price():
    data = request.json
    df = pd.DataFrame([data])
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
    data.pop("make", None)

    df = pd.DataFrame([data])
    prediction = brand_model.predict(df)[0]

    return jsonify({
        "predicted_brand": prediction
    })

# -----------------------------------
# RUN SERVER (Render Compatible)
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
