import streamlit as st
import pandas as pd
import datetime
import uuid
from PIL import Image

from predict import load_trained_model, predict_image, classes

st.set_page_config(
    page_title="Retina AI Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "pred_index" not in st.session_state:
    st.session_state.pred_index = -1
if "pred_label" not in st.session_state:
    st.session_state.pred_label = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "all_probs" not in st.session_state:
    st.session_state.all_probs = []
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "risk_level" not in st.session_state:
    st.session_state.risk_level = ""
if "report_id" not in st.session_state:
    st.session_state.report_id = ""
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #0b3d91 !important;
    }
    p, div, span, label {
        color: #ffffff;
    }
    .hero {
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(135deg, #050505, #111827);
        border: 1px solid #1f2937;
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0;
        color: #0b3d91 !important;
    }
    .hero p {
        margin-top: 8px;
        color: #e5e7eb !important;
    }
    .card {
        background: #0f172a;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #1f2937;
        margin-bottom: 20px;
    }
    .pred-box {
        background: #0f1f34;
        color: #ffffff !important;
        padding: 14px 16px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 14px;
        border: 1px solid #1d4ed8;
    }
    .conf-box {
        background: #111827;
        color: #ffffff !important;
        padding: 14px 16px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 14px;
        border: 1px solid #334155;
    }
    .risk-low {
        background: #062e1f;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #10b981;
    }
    .risk-medium {
        background: #2b2307;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #f59e0b;
    }
    .risk-high {
        background: #2f0b0b;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #ef4444;
    }
    .recommend-box {
        background: #111827;
        color: #ffffff !important;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-top: 12px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        height: 46px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e40af, #1d4ed8);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Smart Retina AI Screening</h1>
    <p>AI-based retinal image classification with session-wise prediction history</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Patient Details")
colp1, colp2, colp3 = st.columns(3)
with colp1:
    patient_name = st.text_input("Patient Name")
with colp2:
    patient_age = st.number_input("Age", min_value=1, max_value=120, step=1)
with colp3:
    scan_date = st.date_input("Scan Date", value=datetime.date.today())

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis")

        if st.button("Analyze Image"):
            pred, confidence, probs = predict_image(model, image)

            st.session_state.prediction_done = True
            st.session_state.pred_index = pred
            st.session_state.pred_label = classes[pred] if pred != 5 else "non_retina"
            st.session_state.confidence = confidence
            st.session_state.all_probs = probs.tolist()
            st.session_state.report_id = f"RPT-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

            if st.session_state.pred_label == "No_DR":
                st.session_state.risk_level = "Low Risk"
                st.session_state.recommendation = "Maintain regular eye checkups and healthy habits."
            elif st.session_state.pred_label in ["Mild", "Moderate"]:
                st.session_state.risk_level = "Medium Risk"
                st.session_state.recommendation = "Consult an eye specialist soon for further evaluation."
            elif st.session_state.pred_label == "non_retina":
                st.session_state.risk_level = "Invalid Image"
                st.session_state.recommendation = "Please upload a proper retinal fundus image."
            else:
                st.session_state.risk_level = "High Risk"
                st.session_state.recommendation = "Immediate medical attention is recommended."

            history_record = {
                "Report ID": st.session_state.report_id,
                "Patient Name": patient_name,
                "Age": patient_age,
                "Scan Date": str(scan_date),
                "Prediction": st.session_state.pred_label,
                "Confidence": f"{st.session_state.confidence * 100:.2f}%",
                "Risk Level": st.session_state.risk_level,
                "Recommendation": st.session_state.recommendation
            }

            st.session_state.prediction_history.append(history_record)

        if st.session_state.prediction_done:
            if st.session_state.pred_index == 5:
                st.error("Please upload a proper retinal fundus image.")
            else:
                st.markdown(
                    f'<div class="pred-box">Prediction: {st.session_state.pred_label}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="conf-box">Confidence: {st.session_state.confidence * 100:.2f}%</div>',
                    unsafe_allow_html=True
                )
                st.progress(float(st.session_state.confidence))

                if st.session_state.risk_level == "Low Risk":
                    st.markdown('<div class="risk-low">Risk Level: Low Risk</div>', unsafe_allow_html=True)
                elif st.session_state.risk_level == "Medium Risk":
                    st.markdown('<div class="risk-medium">Risk Level: Medium Risk</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-high">Risk Level: High Risk</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="recommend-box"><b>Recommendation:</b><br>{st.session_state.recommendation}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.subheader("Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction History CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.success("Prediction history cleared.")
else:
    st.info("No prediction history available yet. Upload and analyze images to build history.")