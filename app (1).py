import streamlit as st
import numpy as np
import pickle
import pytesseract
from PIL import Image
import re

# Load model + scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease AI + OCR", layout="wide")

st.title("🫀 Heart Disease Prediction System")
st.subheader("AI + Medical Report OCR Auto-Fill")

# ---------------- OCR FUNCTION ----------------
def extract_values(text):
    text = text.lower()

    def find_number(key):
        match = re.search(key + r".*?(\d+)", text)
        return int(match.group(1)) if match else 0

    age = find_number("age")
    trestbps = find_number("bp")
    chol = find_number("chol")
    thalach = find_number("heart rate")
    oldpeak = find_number("oldpeak")

    return age, trestbps, chol, thalach, oldpeak


# ---------------- OCR UPLOAD ----------------
st.sidebar.header("📸 Upload Medical Report")

img_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

age = 30
trestbps = 120
chol = 200
thalach = 150
oldpeak = 1.0

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Report", use_container_width=True)

    text = pytesseract.image_to_string(image)

    st.text_area("Extracted Text", text, height=200)

    ocr_age, ocr_bp, ocr_chol, ocr_hr, ocr_oldpeak = extract_values(text)

    age = ocr_age or age
    trestbps = ocr_bp or trestbps
    chol = ocr_chol or chol
    thalach = ocr_hr or thalach
    oldpeak = ocr_oldpeak or oldpeak

    st.success("✅ OCR Auto-Fill Applied")


# ---------------- INPUT UI ----------------
st.sidebar.header("Manual Inputs (Override OCR)")

age = st.sidebar.slider("Age", 1, 100, age)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])

trestbps = st.sidebar.number_input("Resting BP", 80, 200, trestbps)
chol = st.sidebar.number_input("Cholesterol", 100, 600, chol)

fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", ["Yes","No"])
restecg = st.sidebar.selectbox("Resting ECG", [0,1,2])

thalach = st.sidebar.slider("Max Heart Rate", 60, 220, thalach)
exang = st.sidebar.selectbox("Exercise Angina", ["Yes","No"])

oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, float(oldpeak))
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels", [0,1,2,3])

# ---------------- CONVERT ----------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict Risk"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca
    ]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("## 🧾 Result")

    if prediction == 1:
        st.error("🔴 High Risk of Heart Disease")
    else:
        st.success("🟢 No Risk of Heart Disease")

    st.write(f"Probability: {round(prob*100,2)}%")
    st.progress(int(prob*100))

else:
    st.info("Upload report OR enter values and click Predict")
   
       
