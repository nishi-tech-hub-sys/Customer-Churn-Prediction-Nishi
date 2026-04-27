import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Prediction")

st.write("Enter basic customer details:")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict"):

    try:
        input_data = np.array([[tenure, monthly_charges, total_charges]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("Customer is likely to churn ❌")
        else:
            st.success("Customer will stay ✅")

    except:
        st.warning("⚠️ Feature mismatch! Use full dataset features for perfect prediction.")
