import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="DRD", layout="centered")

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

st.markdown("""
<style>
.stApp { background-color: #f5f7fa; }
.block-container { max-width: 900px; }
h1 { text-align: center; color: #0A3D62; }
.card {
    padding: 20px;
    background: white;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:30px;border-radius:15px;
background: linear-gradient(135deg, #1e3c72, #2a5298);
color:white;'>
<h1>🧠 Smart Retina AI Screening</h1>
<p>Single-model retinal vs non-retinal + DR level detection</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

def predict(image):
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        pred, confidence = predict(image)

        if pred == 5:
            st.error("Please upload a retinal fundus image.")
        else:
            label = classes[pred]
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Retina Image", use_container_width=True)

            with col2:
                st.markdown(f"""
                <div class="card">
                <h3>Diagnosis Report</h3>
                <p><b>Stage:</b> {label}</p>
                <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(float(confidence), 1.0))

                if confidence < 0.4:
                    st.warning("Low confidence result. Image may be unclear.")

                st.subheader("Risk Level")
                if label == "No_DR":
                    st.success("Low Risk")
                elif label in ["Mild", "Moderate"]:
                    st.warning("Medium Risk")
                else:
                    st.error("High Risk")

                st.subheader("Recommendation")
                if label == "No_DR":
                    st.write("Maintain regular eye checkups.")
                elif label in ["Mild", "Moderate"]:
                    st.write("Consult an eye specialist soon.")
                else:
                    st.write("Immediate medical attention required!")

                st.info("AI-based screening, not final diagnosis")

st.write("---")
st.markdown("## About")
st.write("- Model: ResNet-18")
st.write("- Dataset: custom 6-class dataset")
st.write("- Classes: 5 DR levels + non_retina")

st.markdown("## Workflow")
st.write("Upload -> AI Model -> Prediction -> Report")