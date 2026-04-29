import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("data/dr_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def generate_gradcam(image):
    img = image.convert("RGB")
    
    # Resize original image to match model input size
    resized_img = img.resize((224, 224))
    rgb_img = np.array(resized_img) / 255.0
    
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    target_layer = model.layer4[-1]
    
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image

st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to detect the DR stage.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        pred, confidence = predict(image)
        st.success(f"Prediction: {class_names[pred]}")
        st.info(f"Confidence: {confidence * 100:.2f}%")

        st.write("### Grad-CAM Heatmap")
        heatmap = generate_gradcam(image)
        st.image(heatmap, caption="Model focus area", use_container_width=True)