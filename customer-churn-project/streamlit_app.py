import streamlit as st
import pandas as pd
import joblib
import logging
import shap
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

explainer = shap.TreeExplainer(model)

# ─────────────────────────────────────────────
# NLP EXPLANATION FUNCTION (NEW)
# ─────────────────────────────────────────────
def generate_nlp_explanation(shap_values, feature_names):
    values = shap_values[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "impact": values
    })

    df["abs"] = df["impact"].abs()
    df = df.sort_values("abs", ascending=False).head(5)

    reasons = []
    for _, row in df.iterrows():
        if row["impact"] > 0:
            reasons.append(f"{row['feature']} increases churn risk")
        else:
            reasons.append(f"{row['feature']} reduces churn risk")

    return reasons


def generate_pdf(probability, prediction):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    p.drawString(100, 750, "Customer Churn Report")
    p.drawString(100, 720, f"Probability: {probability:.2%}")
    p.drawString(100, 700, f"Prediction: {'Churn' if prediction==1 else 'Stay'}")

    p.save()
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Churn Dashboard", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("🚀 Predict Churn"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Feature engineering
    input_data["TotalCharges"] = pd.to_numeric(input_data["TotalCharges"], errors="coerce").fillna(0)
    input_data["avg_monthly_spend"] = input_data["TotalCharges"] / input_data["tenure"].clip(lower=1)
    input_data["charge_delta"] = input_data["MonthlyCharges"] - input_data["avg_monthly_spend"]
    input_data["is_new_customer"] = (input_data["tenure"] <= 6).astype(int)
    input_data["is_long_tenure"] = (input_data["tenure"] >= 48).astype(int)

    service_cols = [
        "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
    ]

    input_data["service_count"] = input_data[service_cols].apply(lambda r: (r == "Yes").sum(), axis=1)
    input_data["has_support"] = ((input_data["OnlineSecurity"] == "Yes") | (input_data["TechSupport"] == "Yes")).astype(int)

    # Encode
    for col, enc in encoders.items():
        input_data[col] = enc.transform(input_data[col].astype(str))

    # Align features
    input_data = input_data[feature_names]

    # Convert common categorical strings to numeric if any remain
    non_numeric = input_data.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        logging.warning("Non-numeric columns before scaling: %s", non_numeric)
        input_data = input_data.replace({"No internet service": "No", "No phone service": "No"})
        input_data = input_data.replace({"Yes": 1, "No": 0})
        non_numeric = input_data.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            logging.error("Non-numeric features remain after replacement: %s", non_numeric)
            st.error(f"Unable to scale: non-numeric features: {non_numeric}")
            st.write(input_data[non_numeric].iloc[0].to_dict())
            raise ValueError(f"Non-numeric features remain: {non_numeric}")

    # Scale
    scaled_data = scaler.transform(input_data)

    # Prediction
    probability = model.predict_proba(scaled_data)[0][1]
    prediction = int(probability >= 0.5)

    st.session_state["probability"] = probability
    st.session_state["prediction"] = prediction

    # ─────────────────────────────────────────────
    # RESULT UI
    # ─────────────────────────────────────────────
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{probability:.2%}")
    col2.metric("Prediction", "Churn" if prediction else "Stay")

    st.progress(int(probability * 100))

    if probability < 0.3:
        st.success("🟢 Low Risk")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # ─────────────────────────────────────────────
    # SHAP
    # ─────────────────────────────────────────────
    st.subheader("🧠 SHAP Explanation")

    shap_values = explainer.shap_values(scaled_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, pd.DataFrame(scaled_data, columns=feature_names), show=False)
    st.pyplot(fig)

    # ─────────────────────────────────────────────
    # NLP EXPLANATION (NEW)
    # ─────────────────────────────────────────────
    st.subheader("🧠 AI Explanation (NLP View)")

    reasons = generate_nlp_explanation(shap_values, feature_names)

    for r in reasons:
        st.write("• " + r)

    st.info(" ".join(reasons))

    # ─────────────────────────────────────────────
    # DOWNLOAD CSV
    # ─────────────────────────────────────────────
    df = pd.DataFrame([{
        "probability": probability,
        "prediction": prediction
    }])

    st.download_button("⬇ CSV Report", df.to_csv(index=False), "report.csv")

    # Generate and offer PDF report
    pdf = generate_pdf(probability, prediction)
    st.download_button("⬇ PDF Report", pdf, "report.pdf", "application/pdf")

    # end of prediction block