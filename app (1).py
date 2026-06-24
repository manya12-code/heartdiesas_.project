import streamlit as st
import pickle
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
keys_to_init = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", 
                "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium"]

for key in keys_to_init:
    if key not in st.session_state:
        
        st.session_state[key] = 0.0 if "creatinine" in key or "platelets" in key else 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction Using Blood Report")
st.markdown("Upload a blood report and predict heart disease risk.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Project Information")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("OCR: Tesseract OCR")
st.sidebar.write("Prediction: Heart Failure Risk")

# ---------------- FILE UPLOAD ----------------
st.subheader("📄 Upload Blood Report")

uploaded_file = st.file_uploader(
    "Choose Report Image",
    type=["png", "jpg", "jpeg"]
)

# Default values
age = 0
sex = 0
platelets = 0
serum_creatinine = 0
serum_sodium = 0

# ---------------- OCR SECTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Report")

    with col2:
    
text=pytesseract.image_image_to_string(img)    
    
if text:  
    import re
    
    
    age_match = re.search(r"Age\s*:\s*(\d+)", text, re.IGNORECASE)
    if age_match:
        st.session_state.age = int(age_match.group(1))
        
    
    platelets_match = re.search(r"Platelets\s*:\s*([\d\.]+)", text, re.IGNORECASE)
    if platelets_match:
        st.session_state.platelets = float(platelets_match.group(1))
    st.rerun()
    st.write("### OCR Extracted Text")
    st.text_area(
            "Extracted Text",
            text,
            height=250
        )

    st.success("OCR Completed Successfully")

# ---------------- INPUT SECTION ----------------
st.subheader("📝 Clinical Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=int(st.session_state.age)
    )

    anaemia = st.selectbox(
        "Anaemia",
        option= [0, 1],
        index= 0 if st.session_state.anaemia==0
                    else 1
    )

    creatinine_phosphokinase = st.number_input(
        "Creatinine Phosphokinase",
        min_value=0,
        value=int(st.session_state.creatinine_phosphokinase)
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1]
    )

    ejection_fraction = st.number_input(
        "Ejection Fraction",
        value=38
    )

with col2:

    high_blood_pressure = st.selectbox(
        "High Blood Pressure",
        [0, 1]
    )

    platelets = st.number_input(
        "Platelets",
        min_value=0.0,
        value=flaot(st.session_state.platelets)
    )

    serum_creatinine = st.number_input(
        "Serum Creatinine",
        value=float(st.session_state.serum_creatinine)
    )

    serum_sodium = st.number_input(
        "Serum Sodium",
        value=int(st.session_state.serum_sodium)
    )

    sex = st.selectbox(
        "Sex",
        [1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    smoking = st.selectbox(
        "Smoking",
        [0, 1]
    )

    time = st.number_input(
        "Follow Up Time",
        value=120
    )

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict Heart Risk"):

    input_data = [[
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time
    ]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    risk_percent = probability[0][1] * 100

    st.subheader("❤️ Prediction Result")

    if prediction[0] == 1:
        st.error(
            f"High Risk of Heart Failure ({risk_percent:.2f}%)"
        )
    else:
        st.success(
            f"Low Risk of Heart Failure ({100-risk_percent:.2f}%)"
        )

    # ---------------- CHART ----------------
    st.subheader("📊 Risk Visualization")

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(
        ["Low Risk", "High Risk"],
        [100-risk_percent, risk_percent]
    )

    ax.set_ylabel("Percentage")
    ax.set_title("Heart Disease Risk")

    st.pyplot(fig)

    # ---------------- SUMMARY ----------------
    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Platelets",
            "Serum Creatinine",
            "Serum Sodium"
        ],
        "Value": [
            age,
            platelets,
            serum_creatinine,
            serum_sodium
        ]
    })

    st.dataframe(summary)
