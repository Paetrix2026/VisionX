import streamlit as st
import pandas as pd
import datetime
import uuid
import io
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

st.set_page_config(
    page_title="Retina AI Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cpu")

# Keep this as 6 because your saved model is still a 6-class model
classes = ["Mild", "Moderate", "No_DR", "Proliferative", "Severe", "non_retina"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "drd_6class.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

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
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

def reset_current_report():
    st.session_state.prediction_done = False
    st.session_state.pred_index = -1
    st.session_state.pred_label = ""
    st.session_state.confidence = 0.0
    st.session_state.all_probs = []
    st.session_state.recommendation = ""
    st.session_state.risk_level = ""
    st.session_state.report_id = ""
    st.session_state.uploaded_image = None
    st.session_state.pdf_data = None

st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #0b3d91 !important;
    }
    p, div, span, label, li {
        color: #ffffff;
    }
    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #050505, #111827);
        border: 1px solid #1f2937;
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0;
        color: #0b3d91 !important;
    }
    .hero p {
        margin: 8px 0 0 0;
        color: #e5e7eb !important;
    }
    .card {
        background: #0f172a;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
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
        margin-top: 10px;
    }
    .risk-medium {
        background: #2b2307;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #f59e0b;
        margin-top: 10px;
    }
    .risk-high {
        background: #2f0b0b;
        color: #ffffff !important;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #ef4444;
        margin-top: 10px;
    }
    .recommend-box {
        background: #111827;
        color: #ffffff !important;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-top: 12px;
    }
    .contact-box {
        background: #111827;
        color: #ffffff !important;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #1d4ed8;
        margin-top: 12px;
        line-height: 1.7;
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
    <p>Prediction, PDF report download, reset option, history tracking, and doctor contact details</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## System Info")
st.sidebar.markdown("""
<div class="card">
<p><b>Model:</b> ResNet-18</p>
<p><b>Classes:</b> 5</p>
<p><b>Inference:</b> CPU</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Doctor / Hospital Contacts")
st.sidebar.markdown("""
<div class="contact-box">
<p><b>1. M. M. Joshi Eye Institute, Sankeshwar</b><br>Phone: 08333-273508</p>
<p><b>2. M. M. Joshi Eye Institute, Sankeshwar</b><br>Mobile: 8550855222</p>
<p><b>3. Sankeshwar Mission Hospital</b><br>Phone: 08333-273426</p>
<p><b>4. Verify before use:</b><br>Murgude Eye Hospital and Laser Centre<br>Phone: +91 8333273508</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Class Order")
for i, c in enumerate(classes):
    st.sidebar.write(f"{i} → {c}")

st.subheader("Patient Details")
colp1, colp2, colp3 = st.columns(3)
with colp1:
    patient_name = st.text_input("Patient Name")
with colp2:
    patient_age = st.number_input("Age", min_value=1, max_value=120, step=1)
with colp3:
    scan_date = st.date_input("Scan Date", value=datetime.date.today())

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

def predict(image):
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        confidence = float(probs[pred])
    return pred, confidence, probs

def generate_pdf_report(image, pred_label, confidence, risk_level, recommendation, all_probs,
                        patient_name, patient_age, scan_date, report_id):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.white)
    pdf.rect(0, 0, width, height, fill=1)

    pdf.setFillColor(colors.HexColor("#0b3d91"))
    pdf.rect(0, height - 80, width, 80, fill=1)

    pdf.setFillColor(colors.white)
    pdf.roundRect(40, height - 65, 42, 42, 8, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#0b3d91"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(61, height - 49, "RA")

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(95, height - 42, "Retina AI Screening Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(95, height - 58, "AI-assisted diabetic retinopathy screening summary")

    y = height - 110
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Patient Details")
    pdf.line(40, y - 5, 250, y - 5)

    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y - 25, f"Patient Name: {patient_name}")
    pdf.drawString(40, y - 45, f"Age: {patient_age}")
    pdf.drawString(40, y - 65, f"Scan Date: {scan_date}")
    pdf.drawString(40, y - 85, f"Report ID: {report_id}")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(320, y, "Prediction Summary")
    pdf.line(320, y - 5, width - 40, y - 5)

    pdf.setFont("Helvetica", 11)
    pdf.drawString(320, y - 25, f"Prediction: {pred_label}")
    pdf.drawString(320, y - 45, f"Confidence: {confidence * 100:.2f}%")
    pdf.drawString(320, y - 65, f"Risk Level: {risk_level}")

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(320, y - 95, "Recommendation:")
    pdf.setFont("Helvetica", 10)

    words = recommendation.split()
    lines = []
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if pdf.stringWidth(test_line, "Helvetica", 10) < 220:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    rec_y = y - 112
    for line in lines:
        pdf.drawString(320, rec_y, line)
        rec_y -= 14

    section_y = y - 135
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, section_y, "Class-wise Probabilities")
    pdf.line(40, section_y - 5, width - 40, section_y - 5)

    pdf.setFont("Helvetica", 10)
    prob_y = section_y - 24
    for class_name, p in zip(classes, all_probs):
        pdf.drawString(50, prob_y, f"{class_name}: {p * 100:.2f}%")
        prob_y -= 14

    img_y = prob_y - 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, img_y, "Uploaded Scan")

    image_reader = ImageReader(image.convert("RGB"))
    pdf.drawImage(image_reader, 40, img_y - 220, width=220, height=200, preserveAspectRatio=True, mask='auto')

    contact_y = 135
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, contact_y, "Doctor / Hospital Contacts")
    pdf.line(40, contact_y - 5, width - 40, contact_y - 5)

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, contact_y - 22, "1. M. M. Joshi Eye Institute, Sankeshwar - 08333-273508")
    pdf.drawString(40, contact_y - 38, "2. M. M. Joshi Eye Institute, Sankeshwar - 8550855222")
    pdf.drawString(40, contact_y - 54, "3. Sankeshwar Mission Hospital - 08333-273426")
    pdf.drawString(40, contact_y - 70, "4. Murgude Eye Hospital and Laser Centre - +91 8333273508")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(380, 78, "________________________")
    pdf.drawString(418, 63, "Authorized Signature")

    pdf.setStrokeColor(colors.grey)
    pdf.line(40, 45, width - 40, 45)
    pdf.setFillColor(colors.grey)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, 30, "This report is generated by an AI screening system and is not a final medical diagnosis.")

    pdf.save()
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = image

    col1, col2 = st.columns([1, 1.15], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis")

        b1, b2 = st.columns(2)

        with b1:
            analyze_clicked = st.button("Analyze Image", use_container_width=True)

        with b2:
            reset_clicked = st.button("Reset Current Report", use_container_width=True)

        if reset_clicked:
            reset_current_report()
            st.success("Current report has been reset.")

        if analyze_clicked:
            pred, confidence, probs = predict(image)

            st.session_state.prediction_done = True
            st.session_state.pred_index = pred
            st.session_state.pred_label = classes[pred]
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

            pdf_buffer = generate_pdf_report(
                image=st.session_state.uploaded_image,
                pred_label=st.session_state.pred_label,
                confidence=st.session_state.confidence,
                risk_level=st.session_state.risk_level,
                recommendation=st.session_state.recommendation,
                all_probs=st.session_state.all_probs,
                patient_name=patient_name,
                patient_age=patient_age,
                scan_date=scan_date,
                report_id=st.session_state.report_id
            )
            st.session_state.pdf_data = pdf_buffer.getvalue()

        if st.session_state.prediction_done:
            if st.session_state.pred_label == "non_retina":
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
                elif st.session_state.risk_level == "Invalid Image":
                    st.markdown('<div class="risk-medium">Risk Level: Invalid Image</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-high">Risk Level: High Risk</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="recommend-box"><b>Recommendation:</b><br>{st.session_state.recommendation}</div>',
                    unsafe_allow_html=True
                )

            if st.session_state.pdf_data is not None:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name=f"{st.session_state.report_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
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
    st.info("No prediction history available yet.")