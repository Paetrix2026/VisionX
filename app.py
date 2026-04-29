import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Retina AI Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cpu")
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
        font-size: 2rem;
    }

    .hero p {
        margin: 6px 0 0 0;
        color: #e5e7eb !important;
    }

    .info-card {
        background: #0f172a;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        border: 1px solid #1f2937;
    }

    .result-card {
        background: #0b1220;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        border: 1px solid #1f2937;
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

    .contact-box {
        background: #111827;
        color: #ffffff !important;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #1d4ed8;
        margin-top: 12px;
        line-height: 1.7;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff !important;
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
    <p>Single-model detection for retinal / non-retinal screening and diabetic retinopathy grading</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## System Info")
st.sidebar.markdown("""
<div class="info-card">
<p><b>Model:</b> ResNet-18</p>
<p><b>Classes:</b> 6</p>
<p><b>Inference:</b> CPU</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Nearby Hospital Contacts")
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

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

def predict(image):
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        confidence = float(probs[pred])
    return pred, confidence, probs

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.15], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Uploaded Fundus Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Analysis")

        if st.button("Analyze Image"):
            pred, confidence, probs = predict(image)
            st.session_state.prediction_done = True
            st.session_state.pred_index = pred
            st.session_state.confidence = confidence
            st.session_state.all_probs = probs.tolist()
            st.session_state.pred_label = "non_retina" if pred == 5 else classes[pred]

        if st.session_state.prediction_done:
            pred = st.session_state.pred_index
            label = st.session_state.pred_label
            confidence = st.session_state.confidence

            if pred == 5:
                st.error("Please upload a retinal fundus image.")
            else:
                st.markdown(f'<div class="pred-box">Prediction: {label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="conf-box">Confidence: {confidence * 100:.2f}%</div>', unsafe_allow_html=True)
                st.progress(float(confidence))

                if label == "No_DR":
                    st.markdown('<div class="risk-low">Risk Level: Low Risk</div>', unsafe_allow_html=True)
                    recommendation = "Maintain regular eye checkups and healthy habits."
                elif label in ["Mild", "Moderate"]:
                    st.markdown('<div class="risk-medium">Risk Level: Medium Risk</div>', unsafe_allow_html=True)
                    recommendation = "Consult an eye specialist soon for further evaluation."
                else:
                    st.markdown('<div class="risk-high">Risk Level: High Risk</div>', unsafe_allow_html=True)
                    recommendation = "Immediate medical attention is recommended."

                st.markdown(f'<div class="recommend-box"><b>Recommendation:</b><br>{recommendation}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Prediction Report", "Confidence Details", "About"])

    with tab1:
        if st.session_state.prediction_done:
            if st.session_state.pred_index == 5:
                st.warning("Uploaded image is predicted as non-retinal.")
            else:
                st.subheader("Detected DR Type")
                st.write(f"**Prediction:** {st.session_state.pred_label}")
                st.write(f"**Interpretation:** The uploaded retinal image is classified as **{st.session_state.pred_label}**.")
        else:
            st.info("Upload an image and click Analyze Image to see the prediction report.")

    with tab2:
        if st.session_state.prediction_done:
            if st.session_state.pred_index != 5:
                st.subheader("Confidence Level")
                st.write(f"**Model Confidence:** {st.session_state.confidence * 100:.2f}%")
                st.progress(float(st.session_state.confidence))

                st.write("### Class-wise Probabilities")
                for i, p in enumerate(st.session_state.all_probs):
                    st.write(f"**{classes[i]}** : {p * 100:.2f}%")
            else:
                st.info("Confidence details are not shown because the image is non-retinal.")
        else:
            st.info("Confidence will appear here after analysis.")

    with tab3:
        st.subheader("About This App")
        st.write("This app uses a single ResNet-18 model for classifying retinal and non-retinal images.")
        st.write("For retinal images, it predicts the DR severity level.")
        st.write("This is an AI-based screening tool and not a final medical diagnosis.")