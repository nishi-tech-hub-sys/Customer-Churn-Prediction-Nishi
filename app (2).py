import streamlit as st
import pickle
import pandas as pd

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Example inputs (adjust based on your dataset)
tenure = st.number_input("Tenure", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Convert input to dataframe
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding to match training columns
input_df = pd.get_dummies(input_df)

# Align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")
