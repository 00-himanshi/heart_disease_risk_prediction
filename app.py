import streamlit as st
import pandas as pd
import joblib


# Model and Scaler

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")


# App Title

heart_icon_url = "https://img.freepik.com/premium-photo/heart-shaped-stickers-3d-hearts-with-different-designs-heart-shape-cartoon-style-stickers-set_1135385-3641.jpg"

st.markdown(
    f"<h1 style='text-align: center;'>"
    f"<img src='{heart_icon_url}' width='40' height='40' style='vertical-align: middle;'> "
    "Heart Disease Risk Prediction"
    "</h1>",
    unsafe_allow_html=True
)

st.write("Enter patient details below to check heart disease risk:")

# User Inputs

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ("Male (1)", "Female (0)"))
chest_pain = st.selectbox(
    "Chest Pain Type",
    ("1 - Typical Angina", "2 - Atypical Angina", "3 - Non-anginal Pain", "4 - Asymptomatic")
)
bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes (1)", "No (0)"))
ekg = st.selectbox(
    "Resting EKG Results",
    ("0 - Normal", "1 - Having ST-T wave abnormality", "2 - Left ventricular hypertrophy")
)
max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
ex_angina = st.selectbox("Exercise Induced Angina", ("Yes (1)", "No (0)"))
st_depression = st.number_input("ST Depression Induced by Exercise (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the ST Segment", ("1 - Upsloping", "2 - Flat", "3 - Downsloping"))
vessels = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0–3)", 0, 3, 0)
thallium = st.selectbox(
    "Thallium Test Result",
    ("3 - Normal", "6 - Fixed Defect", "7 - Reversible Defect")
)

# ==========================
# Convert to Numeric Values
# ==========================
sex_val = int(sex.split('(')[1].replace(')', ''))
cp_val = int(chest_pain.split('-')[0].strip())
fbs_val = int(fbs.split('(')[1].replace(')', ''))
ekg_val = int(ekg.split('-')[0].strip())
ex_angina_val = int(ex_angina.split('(')[1].replace(')', ''))
slope_val = int(slope.split('-')[0].strip())
thallium_val = int(thallium.split('-')[0].strip())

# ==========================
# Create DataFrame for Model
# ==========================
input_data = pd.DataFrame([[
    age, sex_val, cp_val, bp, chol, fbs_val, ekg_val,
    max_hr, ex_angina_val, st_depression, slope_val, vessels, thallium_val
]], columns=[
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium"
])

# ==========================
# Apply Same Scaling as Training
# ==========================
input_scaled = scaler.transform(input_data)

# ==========================
# Prediction Button
# ==========================
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(
            f"""
            <div style="
                background-color: #FFCDD2;
                border-left: 8px solid #B71C1C;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
            ">
                <h3 style='color: #B71C1C;'>⚠️ High Risk of Heart Disease</h3>
                <p style='color: #880E4F;'>Predicted probability: <strong>{probability:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color: #C8E6C9;
                border-left: 8px solid #1B5E20;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
            ">
                <h3 style='color: #1B5E20;'>✅ Low Risk of Heart Disease</h3>
                <p style='color: #2E7D32;'>Predicted probability: <strong>{probability:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================
# Footer
# ==========================
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>⚕️ This app uses a trained Logistic Regression model to predict heart disease risk.</p>",
    unsafe_allow_html=True
)
