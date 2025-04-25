import streamlit as st
import base64
import pandas as pd
import os
import requests 
import tempfile

# --- Page Config ---
st.set_page_config(page_title="Patient Form", layout="wide")

# --- Custom CSS for Black Background and White Form Block ---
custom_css = """
<style>
.stApp {
    background-color: black !important;
    color: white !important;
}

.stFileUploader>div {
    background: #f0f0f0 !important;
    padding: 10px;
    border-radius: 8px;
}
.stButton>button {
    background-color: #00C896 !important;
    color: black !important;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #111 !important;
    color: white !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Form UI ---
st.markdown('<h1 style="color: #00C896; text-align: center;">🧑‍⚕️Patient Credentials Form</h1>', unsafe_allow_html=True)

# Input Fields
patient_name = st.text_input("Full Name*")
age = st.number_input("Age*", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
phone = st.text_input("Phone Number")
date = st.date_input("Date of Visit")
medications = st.text_area("Medications (if any)")
history = st.text_area("Previous Medical History")
eeg_file = st.file_uploader("Upload EEG File*", type=["mat"])

# On Submit
if st.button("Submit"):
    if patient_name and age and eeg_file:
        try:
            # --- Save EEG file temporarily ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                tmp.write(eeg_file.getbuffer())
                tmp_path = tmp.name

            # --- Post to FastAPI endpoint ---
            with open(tmp_path, "rb") as f:
                files = {"file": (eeg_file.name, f, "application/octet-stream")}
                response = requests.post(" https://8d23-203-82-54-66.ngrok-free.app/run_inference/", files=files)  # Replace with your actual API URL

            # --- Handle response ---
            if response.status_code == 200:
                result = response.json()
                predictions = result["predictions"]  # ✅ Save the predictions in a variable

                # Store predictions in session state (optional)
                st.session_state["patient_data"] ={
            "name": patient_name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "date": str(date),
            "medications": medications,
            "history": history,
            "eeg_file": eeg_file
        }


                st.success("✅ Inference successful!")
                st.switch_page("pages/report_page.py")  # Optional navigation

            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"❌ Exception occurred: {e}")
    else:
        st.error("❌ Please fill in all required fields and upload an EEG file.")

st.markdown('</div>', unsafe_allow_html=True)
