# Customer Churn Prediction Dashboard

AI-powered Customer Churn Prediction web application built using:

* Python
* XGBoost / Scikit-learn
* Streamlit
* SHAP Explainability
* Docker
* Render Deployment

---

# Features

## Customer Churn Prediction

Predict whether a telecom customer is likely to churn.

## SHAP Explainability

Shows why the model predicts churn.

## Interactive Dashboard

Modern dark-themed Streamlit UI.

## Download Reports

* PDF Report
* CSV Report

## NLP Customer Risk Analysis

Generates customer retention insights.

## Dockerized Deployment

Fully containerized using Docker.

---

# Tech Stack

| Technology   | Usage                  |
| ------------ | ---------------------- |
| Python       | Backend Logic          |
| Streamlit    | Frontend UI            |
| XGBoost      | Machine Learning Model |
| Scikit-learn | ML Utilities           |
| SHAP         | Explainable AI         |
| Docker       | Containerization       |
| Render       | Cloud Deployment       |

---

# Project Structure

```bash
customer-churn-project/
│
├── streamlit_app.py
├── app.py
├── requirements.txt
├── Dockerfile
├── churn_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── feature_names.pkl
├── churn_model_results.png
│
├── .streamlit/
│   └── config.toml
│
├── logs/
│   └── app.log
│
└── README.md
```

---

# Machine Learning Pipeline

## Data Preprocessing

* Missing value handling
* Feature engineering
* Label encoding
* Standard scaling

## Feature Engineering

Additional engineered features:

* avg_monthly_spend
* charge_delta
* is_new_customer
* is_long_tenure
* service_count
* has_support

## Model Training

* XGBoost Classifier
* RandomizedSearchCV
* Stratified K-Fold Cross Validation

---

# Model Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | ~80%  |
| AUC-ROC  | ~0.84 |

---

# Run Locally

## Clone Repository

```bash
git clone https://github.com/Ghanashyamgudela/Customer-Churn-Prediction.git
```

## Navigate to Project

```bash
cd Customer-Churn-Prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t churn-dashboard .
```

## Run Docker Container

```bash
docker run -p 8501:8501 churn-dashboard
```

## Open Application

```text
http://localhost:8501
```

---

# Render Deployment

## Deployment Steps

1. Push project to GitHub
2. Create new Web Service on Render
3. Select Docker environment
4. Set Dockerfile path:

```text
Dockerfile
```

5. Deploy application

---

# SHAP Explainability

SHAP (SHapley Additive exPlanations) helps explain:

* Why a customer may churn
* Important contributing features
* Feature impact on predictions

---

# Future Improvements

* Authentication system
* Database integration
* Real-time API
* Customer retention recommendations
* Email alerts
* Cloud monitoring

---

# Author

Ghanashyam Gudela

---

# License

This project is for educational and portfolio purposes.
