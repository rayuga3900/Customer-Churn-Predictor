import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__) 
MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.title("Churn Predictor")

num_cols = ['tenure', 'MonthlyCharges','TotalCharges']  
cat_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
           'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
           'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

# Single prediction - Left column
col1, col2 = st.columns(2)

with col1:
    st.header("Single Prediction")
    
    # Numeric inputs
    for col in num_cols:
        st.number_input(col, key=col)
        st.markdown("---")
    
    # Categorical inputs options
    cat_options = {
        'gender': ['Female', 'Male'],
        'SeniorCitizen': ['0', '1'],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
    }
    
    for col in cat_cols: #as col[label] is mapped to options
        st.selectbox(col, cat_options[col], key=col)
        st.markdown("---")
    
    if st.button(" Predict Churn", key="single"):
        input_data = {col: st.session_state[col] for col in num_cols + cat_cols}
        sample = pd.DataFrame([input_data])
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]
        st.success(f"**Churn**: {'Yes ' if prediction == 1 else 'No '}")
        st.info(f"**Probability**: {probability:.1%}")

# Batch prediction - Right column
with col2:
    st.header(" Batch Prediction")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="uploader")
    if st.button("Predict Batch", key="batch_predict"):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required_cols = num_cols + cat_cols
            
            if not all(col in df.columns for col in required_cols):
                st.error(f" Missing columns: {set(required_cols) - set(df.columns)}")
            else:
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:,1]
                
                df["Churn"] = ["Yes " if p==1 else "No " for p in predictions]
                df["Probability"] = probabilities
                
                st.success(" Batch predictions completed!")
                st.dataframe(df, use_container_width=True)
                
                churn_rate = (predictions == 1).mean()
                st.metric("Churn Rate", f"{churn_rate:.1%}")

