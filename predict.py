import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# --- Dataset for test images (using .png) ---
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # ID from test.csv
        img_path = os.path.join(self.img_dir, f"{img_name}.png")  # use .png
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --- Test dataset and loader ---
test_ds = TestDataset("data/test.csv", "data/test_images", transform)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


# --- Device and model (same as train.py: resnet18) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)  # no pretrained weights
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes: 0–4
model.load_state_dict(torch.load("data/dr_model.pth", map_location=device))
model = model.to(device)
model.eval()


# --- Class names ---
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
predictions = []


# --- Make predictions ---
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for name, pred in zip(img_names, preds):
            predictions.append([name, pred.item(), class_names[pred.item()]])


# --- Save predictions ---
pred_df = pd.DataFrame(predictions, columns=["image_id", "predicted_class", "predicted_label"])
pred_df.to_csv("data/test_predictions.csv", index=False)


print("Predictions saved to data/test_predictions.csv")
print(pred_df.head())