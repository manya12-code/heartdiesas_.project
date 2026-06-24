import streamlit as st
import numpy as np
import pickle
from PIL import Image
import pytesseract
import re

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Failure Prediction", layout="wide")

st.title("🏥 Heart Failure Risk Prediction System")
st.markdown("### AI-based Medical Risk Analyzer")

# ---------------- OCR (ONLY AGE OPTIONAL) ----------------
def extract_age(text):
    text = text.lower()
    match = re.search(r"age\D*(\d+)", text)
    return int(match.group(1)) if match else None


# ---------------- SIDEBAR ----------------
st.sidebar.header("📋 Patient Input Form")

img_file = st.sidebar.file_uploader("📸 Upload Report (Optional)", type=["png", "jpg", "jpeg"])

age = 50  # default fallback

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Report")

    text = pytesseract.image_to_string(image)
    st.text_area("Extracted Text", text, height=200)

    ocr_age = extract_age(text)
    if ocr_age is not None:
        age = ocr_age
        st.success(f"✅ Age extracted: {age}")


# ---------------- INPUTS ----------------
age = st.sidebar.slider("Age", 1, 100, age)

anaemia = st.sidebar.selectbox("Anaemia", ["No", "Yes"])
creatinine_phosphokinase = st.sidebar.number_input("Creatinine Phosphokinase", 0, 8000, 100)
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
ejection_fraction = st.sidebar.slider("Ejection Fraction", 10, 80, 38)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
platelets = st.sidebar.number_input("Platelets", 10000, 1000000, 250000)
serum_creatinine = st.sidebar.number_input("Serum Creatinine", 0.1, 10.0, 1.0)
serum_sodium = st.sidebar.slider("Serum Sodium", 100, 150, 137)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
time = st.sidebar.slider("Follow-up Time (days)", 0, 300, 100)

# ---------------- CONVERT ----------------
anaemia = 1 if anaemia == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Yes" else 0


# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict Risk"):

    input_data = np.array([[
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
    ]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("## 🧾 Result")

    if prediction == 1:
        st.error("🔴 High Risk of Heart Failure")
    else:
        st.success("🟢 Low Risk of Heart Failure")

    st.write(f"### Probability: {round(prob*100, 2)}%")
    st.progress(int(prob * 100))

    st.markdown("---")
    st.info("⚠️ Disclaimer: This is an AI prediction tool, not a medical diagnosis.")
else:
    st.info("👈 Enter patient data and click Predict")
  
