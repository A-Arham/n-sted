# 🧠 N-STED: Single Trial ERP Analysis for MDD Diagnosis

## 📌 Overview

**N-STED** is an interactive **Streamlit-based web application** that assists researchers and doctors in analyzing EEG data for **single-trial ERP extraction** using advanced deep learning models. The application is designed to aid in the **diagnosis of Major Depressive Disorder (MDD)**.

---

## 🚀 Features

- 🎨 **Modern, responsive UI** with custom styling
- 📁 **EEG `.mat` file uploader** for ERP analysis
- 🧑‍⚕️ **Patient data form** with age, gender, medical history, etc.
- 🧠 **Single-trial ERP inference via FastAPI**
- 📄 **Dynamic report generation** in PDF format
- 📥 **Downloadable ERP reports** for clinical record keeping

---

## 🧩 Application Workflow

### 1. `app.py` (Homepage)
- Displays welcome message and description.
- Loads background and image via base64.
- Redirects to `patient_page.py` on **"Get Started"** button click.

### 2. `patient_page.py` (Patient Form)
- User fills in patient information and uploads `.mat` EEG file.
- Submits the form, which:
  - Sends EEG file to **FastAPI endpoint** `/run_inference/`
  - Saves patient data to `st.session_state`
  - Redirects to `report_page.py` after successful inference

### 3. `report_page.py` (Report Viewer)
- Loads patient data from session.
- Calls API at `/predict-from-saved/` to get predicted class (e.g., "Normal", "MDD").
- Displays a styled summary of patient and inference result.
- Allows user to:
  - **Generate PDF report**
  - **Download report**
  - **Create new patient entry**

---

## 🔧 Dependencies

Make sure the following Python packages are installed:

```bash
pip install streamlit pandas requests fpdf
