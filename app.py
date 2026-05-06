import streamlit as st
import pandas as pd
import joblib



model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# PAGE TITLE


st.title("Customer Churn Prediction")

st.write("Enter customer information below:")


# USER INPUTS


SeniorCitizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

tenure = st.slider("Tenure (months)", 0, 72, 12)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

Dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

PhoneService = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)


# PREDICT BUTTON


if st.button("Predict Churn"):

    # RAW INPUT DATAFRAME
    input_data = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'gender': [gender],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod]
    })


    # APPLY SAME ENCODING
    

    input_data = pd.get_dummies(input_data, drop_first=True)

    # ADD MISSING COLUMNS
   

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # ENSURE SAME COLUMN ORDER
    

    input_data = input_data[feature_columns]

   
    # PREDICT
   

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # SHOW RESULTS
    

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")

    st.write(f"Churn Probability: {probability:.2f}")