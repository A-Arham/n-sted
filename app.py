import streamlit as st
import base64

# --- Page Config ---
st.set_page_config(page_title="N-STED", layout="wide")

# Function to encode the image in base64 format
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert "brain.tif" to base64
image_base64 = get_base64_image("brainglowy2.png")

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #111;  /* Dark background */
        color: white; /* Default Text Color */
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 80%;
    }}
    .text-container {{
        max-width: 50%;
        text-align: left;
    }}
    .title {{
        font-size: 100px;
        font-weight: bold;
        margin-bottom: 16px;
        color: #00C896; /* Green */
    }}
    .subtitle {{
        font-size: 35px;
        font-weight: bold;
        color: white;
    }}
    .description {{
        font-size: 25px;
        color: #DDDDDD; /* Light Gray for readability */
    }}
    .highlight {{
        color: #00C896; /* Green Highlight */
        font-weight: bold;
    }}

    /* Button with Black Hover Effect */
    .stButton>button {{
     background-color: #00C896 !important;
     color: black !important;
     padding: 15px 30px !important; /* Adjusted button size */
     font-size: 20px !important;
     font-weight: bold !important;
     border: none !important;
     border-radius: 10px !important;
     cursor: pointer !important;
     width: 35% !important;
     transition: background-color 0.3s ease-in-out !important;
     margin-top: 15px !important;  /* Increased space between button and description */
    }}

    .stButton>button:hover {{
        background-color: #000000 !important; /* black on hover */
        color: white !important; /* Ensure text stays visible */
    }} 
    </style>
    """,
    unsafe_allow_html=True
)

# --- Page Layout ---
st.markdown('<div class="container">', unsafe_allow_html=True)

# Left Column - Text & Button
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="text-container">', unsafe_allow_html=True)
    st.markdown('<div class="title">N-STED</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Single Trial ERP Analysis for MDD Diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="description">
        This application allows researchers/doctors to upload 
        <span class="highlight">EEG data files</span>, process them and <span class="highlight"> extract single trial ERP</span> using advanced 
        deep learning models, and diagnoses <span class="highlight">Major Depressive Disorder (MDD)</span> 
        in patients.
        </p>
        """,
        unsafe_allow_html=True
    )

    # "Get Started" Button (Navigates to Upload Page)
    if st.button("Get Started →"):
        st.switch_page("pages/patient_page.py")  # Ensure patient_page.py exists in "pages/"
    
    st.markdown('</div>', unsafe_allow_html=True)
  
# Right Column - Display Large Image Using Base64 Encoding
with col2:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{image_base64}" style="width: 100%; max-width: 450px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
