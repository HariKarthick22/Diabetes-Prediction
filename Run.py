import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource
def load_models():
    try:
        model = load_model('/Users/harikarthick/Desktop/Health /cnn_feature_extractor.h5', compile=False)
        scaler = joblib.load('/Users/harikarthick/Desktop/Health /diabetes_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler:\n{e}")
        return None, None

model, scaler = load_models()
if model is None or scaler is None:
    st.stop()

st.set_page_config(page_title="Diabetes Predictor with Bulk Upload", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.markdown("""
Enter patient health data manually or upload a CSV file to predict diabetes risk in bulk.

**CSV must have these columns in this order:**  
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`
""")

# Sidebar toggle between single and bulk prediction
mode = st.sidebar.radio("Select mode:", ["Single Prediction", "Bulk Prediction"])

if mode == "Single Prediction":
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 2)
        glucose = st.slider("Glucose (mg/dL)", 50, 200, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 30, 130, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

    with col2:
        insulin = st.slider("Insulin (μU/mL)", 0, 850, 100)
        bmi = st.slider("BMI", 10.0, 70.0, 30.5)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.45)
        age = st.slider("Age", 10, 100, 25)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    if st.button("🧠 Predict Single Patient"):
        try:
            input_scaled = scaler.transform(input_data)
            pred_prob = model.predict(input_scaled)[0][0]
            prediction = int(pred_prob > 0.5)
            confidence = pred_prob if prediction == 1 else 1 - pred_prob

            st.divider()
            st.subheader("🧪 Prediction Result")

            if prediction == 0:
                st.error(f"🔴 High Diabetes Risk Detected")
                st.metric("Confidence", f"{confidence * 100:.2f}%", delta="⚠️ Risk")
            else:
                st.success(f"🟢 Low Diabetes Risk")
                st.metric("Confidence", f"{confidence * 100:.2f}%", delta="✅ Safe")

        except Exception as e:
            st.error(f"⚠️ Prediction Error:\n{e}")

elif mode == "Bulk Prediction":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            if list(df.columns) != expected_cols:
                st.error(f"❌ CSV columns do not match expected columns:\n{expected_cols}")
            else:
                # Scale input features
                input_scaled = scaler.transform(df.values)

                # Predict probabilities
                preds_prob = model.predict(input_scaled).flatten()
                preds = (preds_prob > 0.5).astype(int)

                df['Diabetes_Prediction'] = preds
                df['Risk_Confidence'] = np.where(preds == 1, preds_prob, 1 - preds_prob)

                st.success("✅ Bulk Prediction Complete!")
                st.dataframe(df)

                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download predictions as CSV",
                    data=csv,
                    file_name='diabetes_predictions.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"⚠️ Error processing uploaded file:\n{e}")

