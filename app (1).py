import streamlit as st
import numpy as np
import pickle
import pytesseract
from PIL import Image
import re

# --- STEP 1: INITIALIZE SESSION STATE FOR ALL VARIABLES ---
# Isse app crash ya rerun hone par data delete nahi hoga
ml_keys = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", 
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium"
]

for key in ml_keys:
    if key not in st.session_state:
        # Floats ke liye 0.0 baaki binary/integers ke liye 0 default value
        st.session_state[key] = 0.0 if "creatinine" in key or "platelets" in key else 0

# --- STEP 2: LOAD MACHINE LEARNING MODEL ---
# Apne pickle model file ka sahi path check kar lein
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    st.error("Error: 'model.pkl' file nahi mili! Base directory me model file jodein.")

# --- STEP 3: APP UI HEADER DESIGN ---
st.title("❤️ Heart Disease Prediction Using Blood Report")
st.write("Upload a blood report and predict heart disease risk.")

# --- STEP 4: OCR SECTION (BLOOD REPORT UPLOAD) ---
st.subheader("📄 Upload Blood Report")
uploaded_file = st.file_uploader("Choose Report Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Spinner wrapper inside proper block indentation
    with st.spinner("Extracting text from image..."):
        # Fixed trailing Attribute errors and name definition parameters
        text = pytesseract.image_to_string(img)
        
        if text:
            # Regex templates matching common blood report labels
            age_match = re.search(r"Age\s*:\s*(\d+)", text, re.IGNORECASE)
            if age_match:
                st.session_state.age = int(age_match.group(1))
                
            anaemia_match = re.search(r"Anaemia\s*:\s*(\d+)", text, re.IGNORECASE)
            if anaemia_match:
                st.session_state.anaemia = int(anaemia_match.group(1))

            cpk_match = re.search(r"Phosphokinase\s*:\s*(\d+)", text, re.IGNORECASE)
            if cpk_match:
                st.session_state.creatinine_phosphokinase = int(cpk_match.group(1))

            diabetes_match = re.search(r"Diabetes\s*:\s*(\d+)", text, re.IGNORECASE)
            if diabetes_match:
                st.session_state.diabetes = int(diabetes_match.group(1))

            bp_match = re.search(r"Blood\s*Pressure\s*:\s*(\d+)", text, re.IGNORECASE)
            if bp_match:
                st.session_state.high_blood_pressure = int(bp_match.group(1))

            platelets_match = re.search(r"Platelets\s*:\s*([\d\.]+)", text, re.IGNORECASE)
            if platelets_match:
                st.session_state.platelets = float(platelets_match.group(1))

            sc_match = re.search(r"Serum\s*Creatinine\s*:\s*([\d\.]+)", text, re.IGNORECASE)
            if sc_match:
                st.session_state.serum_creatinine = float(sc_match.group(1))

            sodium_match = re.search(r"Sodium\s*:\s*(\d+)", text, re.IGNORECASE)
            if sodium_match:
                st.session_state.serum_sodium = int(sodium_match.group(1))

            st.success("Data successfully extracted! Inputs fields below have been updated.")
            st.rerun()

# --- STEP 5: DYNAMIC INPUT FIELDS (CONNECTED TO STATE) ---
st.subheader("🩺 Patient Health Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=0, max_value=120, value=int(st.session_state.age)
    )
    anaemia = st.selectbox(
        "Anaemia", options=["No", "Yes"], index=0 if st.session_state.anaemia == 0 else 1
    )
    creatinine_phosphokinase = st.number_input(
        "Creatinine Phosphokinase (CPK)", min_value=0, value=int(st.session_state.creatinine_phosphokinase)
    )
    diabetes = st.selectbox(
        "Diabetes", options=["No", "Yes"], index=0 if st.session_state.diabetes == 0 else 1
    )

with col2:
    high_blood_pressure = st.selectbox(
        "High Blood Pressure", options=["No", "Yes"], index=0 if st.session_state.high_blood_pressure == 0 else 1
    )
    platelets = st.number_input(
        "Platelets Count", min_value=0.0, value=float(st.session_state.platelets)
    )
    serum_creatinine = st.number_input(
        "Serum Creatinine Level", min_value=0.0, value=float(st.session_state.serum_creatinine)
    )
    serum_sodium = st.number_input(
        "Serum Sodium Level", min_value=0, value=int(st.session_state.serum_sodium)
    )

# --- STEP 6: PREDICTION WORKFLOW ---
st.markdown("---")
if st.button("Predict Heart Failure Risk", type="primary"):
    if model is not None:
        # Text/Dropdown values map into Model Binary Matrix (Yes=1, No=0)
        anaemia_val = 1 if anaemia == "Yes" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        hbp_val = 1 if high_blood_pressure == "Yes" else 0
        
        # Features array sequencing according to model training architecture
        features = np.array([[
            age, anaemia_val, creatinine_phosphokinase, diabetes_val, 
            high_blood_pressure_val, platelets, serum_creatinine, serum_sodium
        ]])
        
        prediction = model.predict(features)
        
        # Result analysis block display
        if prediction[0] == 1:
            st.error("⚠️ High Risk: Report suggests tendencies towards cardiovascular vulnerability.")
        else:
            st.success("✅ Low Risk: Patient features map standard range benchmarks.")
    else:
        st.warning("Model file not configured properly. Cannot trigger prediction pipeline.")
   
       
