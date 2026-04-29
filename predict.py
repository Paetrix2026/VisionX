import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

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

def load_trained_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image: Image.Image):
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        confidence = float(probs[pred])
    return pred, confidence, probs