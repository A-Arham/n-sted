import streamlit as st
from fpdf import FPDF
import requests


# --- Page Configuration ---
st.set_page_config(page_title="Report Page", layout="wide")

# --- Custom CSS for Black Background and Styled Components ---
st.markdown("""
<style>
.stApp {
    background-color: black !important;
    color: white !important;
}
.title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #00C896;
    margin-bottom: 20px;
}
.stButton > button,
.stDownloadButton > button {
    font-size: 35px;
    font-weight: bold;
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
    background-color: #00C896 !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    width: 25% !important;
    transition: background-color 0.3s;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #000 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- Page Title ---
st.markdown('<div class="title">Processing complete!</div>', unsafe_allow_html=True)

# --- Check Session State ---
if "patient_data" not in st.session_state:
    st.warning("No patient data found. Please fill out the form first.")
    st.stop()

# --- Extract Patient Data ---
data = st.session_state["patient_data"]

# --- Generate PDF Function ---
def create_pdf(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_text_color(0, 200, 150)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, " Patient's ERP Report & Performa", ln=True, align="C")
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)

    for line in content.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            pdf.set_text_color(0, 200, 150)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(40, 10, f"{key.strip()}:", ln=0)

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, value.strip())
            pdf.ln(1)
        else:
            pdf.multi_cell(0, 10, line)
            pdf.ln(1)

    return pdf.output(dest="S").encode("latin1")

# --- Generate Report Button ---
if st.button("Generate Report"):
    # --- Trigger the API ---
    api_url = " https://8d23-203-82-54-66.ngrok-free.app/predict-from-saved/"  # Replace with your actual FastAPI host/port if different
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        result_data = response.json()
        prediction_successful = True
        prediction_class = result_data.get("predicted_class", "Unknown")
    except Exception as e:
        prediction_successful = False
        prediction_class = str(e)

    # --- Print result for verification ---
    st.write("Prediction Successful:", prediction_successful)
    st.write("Predicted Class:", prediction_class)

    # --- Styled Report Summary ---
    styled_report = f"""
    <div style="display: flex; justify-content: center; margin-top: 30px;">
        <div style="background-color: white; color: black; padding: 20px 30px; border-radius: 10px; width: 65%; font-size: 15px;">
            <p><strong style="color:#00C896;">Patient Name:</strong> {data['name']}</p>
            <p><strong style="color:#00C896;">Age:</strong> {data['age']}</p>
            <p><strong style="color:#00C896;">Gender:</strong> {data['gender']}</p>
            <p><strong style="color:#00C896;">Phone:</strong> {data['phone']}</p>
            <p><strong style="color:#00C896;">Visit Date:</strong> {data['date']}</p>
            <p><strong style="color:#00C896;">Medications:</strong> {data['medications'] or 'None'}</p>
            <p><strong style="color:#00C896;">Medical History:</strong> {data['history'] or 'None'}</p>
            <p><strong style="color:#00C896;">EEG Analysis:</strong> {prediction_class}</p>
            <p><strong style="color:#00C896;">Conclusion:</strong> {'No signs of mental disorder' if prediction_class == 'Normal' else 'Signs of MDD detected'}</p>
        </div>
    </div>
    """
    st.markdown(styled_report, unsafe_allow_html=True)

    # --- Prepare Report Text for PDF ---
    report_content = f"""
    Patient Name: {data['name']}
    Age: {data['age']}
    Gender: {data['gender']}
    Phone: {data['phone']}
    Visit Date: {data['date']}
    Medications: {data['medications'] or 'None'}
    Medical History: {data['history'] or 'None'}
    EEG Analysis: {prediction_class}
    Conclusion: {'No signs of mental disorder' if prediction_class == 'Normal' else 'Signs of MDD detected'}
    """

    pdf_bytes = create_pdf(report_content)

    # --- Space ---
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Download Report Button ---
    st.download_button(
        label="Download Report",
        data=pdf_bytes,
        file_name=f"{data['name'].replace(' ', '_')}_report.pdf",
        mime="application/pdf"
    )


# --- New Patient Button ---
new_patient = st.button("Create New Patient")
if new_patient:
    st.switch_page("pages/patient_page.py")
