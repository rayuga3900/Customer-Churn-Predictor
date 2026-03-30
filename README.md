# Telco Customer Churn Prediction App
<div>
<img width="1699" height="899" alt="Screenshot 2026-01-04 144444" src="https://github.com/user-attachments/assets/86c09a31-d2d6-4cb9-8cd2-984761c8702f" />
<img width="1683" height="858" alt="Screenshot 2026-01-04 144554" src="https://github.com/user-attachments/assets/b1561ba8-8bbc-4c1e-b036-28831fd6c2c2" />
</div>

This project predicts whether a telecom customer is likely to churn based on their service usage and account information.  
It includes a machine learning pipeline, a Streamlit web app, and a Docker setup for deployment.

---

## Overview

Customer churn is a major business problem in the telecom industry.  
This project aims to predict churn using customer demographics, service subscriptions, and billing details.

The model is trained on the Telco Customer Churn dataset and deployed as an interactive web application.

---

## Features Used

### Numeric Features
- Tenure
- Monthly Charges
- Total Charges

### Categorical Features
- Gender
- Senior Citizen
- Partner / Dependents
- Phone & Internet services
- Online services (Security, Backup, Streaming, etc.)
- Contract type
- Payment method
- Paperless billing

---

## Model Details

- **Algorithm:** Random Forest Classifier  
- **Preprocessing:**
  - Mean imputation for numeric features
  - Most frequent imputation for categorical features
  - One-hot encoding for categorical variables
- **Class imbalance handling:** Balanced class weights
- **Pipeline:** End-to-end Scikit-learn pipeline

---

## Model Evaluation

The model is evaluated using:
- Accuracy
- ROC-AUC score
- Confusion matrix

ROC-AUC is used as the primary metric due to class imbalance.

---

## Streamlit Application

The app supports:
- Single customer churn prediction
- Batch prediction using CSV upload
- Churn probability output
- Overall churn rate calculation

---

## Running Locally
```bash
Install dependencies:
pip install -r requirements.txt


Run the app:
streamlit run app/app.py

Running with Docker

Build the image:
docker build -t telco-churn-app .


Run the container:
docker run -p 5000:8501 telco-churn-app


Open:
http://localhost:5000
