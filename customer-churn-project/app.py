from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from pyparsing import col
import logging
import mysql.connector
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a"
)

logger = logging.getLogger(__name__) 

# ─────────────────────────────────────────────────────────────
# Load saved artifacts
# ─────────────────────────────────────────────────────────────

# Load saved files

model = joblib.load("churn_model.pkl")
logger.info("Model loaded successfully")

scaler = joblib.load("scaler.pkl")
logger.info("Scaler loaded successfully")

encoders = joblib.load("label_encoders.pkl")
logger.info("Encoders loaded successfully")

feature_names = joblib.load("feature_names.pkl")
logger.info("Feature names loaded successfully")

# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# MySQL Connection
# ─────────────────────────────────────────────────────────────

db = mysql.connector.connect(
    host="host.docker.internal",
    user="root",
    password="Ghana@1230",
    database="churn_db"
)

cursor = db.cursor()

print("MySQL Connected Successfully!")
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Home Route
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():

    return render_template("index.html")


# ─────────────────────────────────────────────────────────────
# Preprocessing Function
# ─────────────────────────────────────────────────────────────

def preprocess_input(data):

    df = pd.DataFrame([data])

    # Convert numeric column
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    # Feature Engineering
    df["avg_monthly_spend"] = (
        df["TotalCharges"] /
        df["tenure"].clip(lower=1)
    )

    df["charge_delta"] = (
        df["MonthlyCharges"] -
        df["avg_monthly_spend"]
    )

    df["is_new_customer"] = (
        df["tenure"] <= 6
    ).astype(int)

    df["is_long_tenure"] = (
        df["tenure"] >= 48
    ).astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["service_count"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(),
        axis=1
    )

    df["has_support"] = (
        (df["OnlineSecurity"] == "Yes") |
        (df["TechSupport"] == "Yes")
    ).astype(int)

    # Encode categorical columns
    for col in encoders:

        if col in df.columns:

            encoder = encoders[col]

            df[col] = df[col].apply(
                lambda x: (
                    x if x in encoder.classes_
                    else encoder.classes_[0]
                )
            )

            df[col] = encoder.transform(
                df[col].astype(str)
            )

# Match feature order
    df = df[feature_names]

# Convert all columns to numeric
    df = df.apply(pd.to_numeric)

# Scale features
    scaled = scaler.transform(df)

    return scaled
    # ─────────────────────────────────────────────────────────────
    # Prediction Route
    # ─────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        processed_data = preprocess_input(data)

        probability = model.predict_proba(
            processed_data
        )[0][1]

        prediction = int(probability >= 0.5)

        # Save to MySQL
        sql = """
        INSERT INTO predictions (
            gender,
            tenure,
            monthly_charges,
            total_charges,
            churn_probability,
            prediction
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (
            data["gender"],
            int(data["tenure"]),
            float(data["MonthlyCharges"]),
            float(data["TotalCharges"]),
            float(probability),
            int(prediction)
        )

        cursor.execute(sql, values)

        db.commit()

        result = {
            "churn_probability": round(float(probability), 4),
            "prediction": prediction,
            "message": (
                "Customer likely to churn"
                if prediction == 1
                else
                "Customer likely to stay"
            )
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# ─────────────────────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )