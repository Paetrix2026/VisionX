import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# --- Dataset class ---
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]                     # use precomputed path from CSV
        label = row["diagnosis"]                        # 0–4
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# --- Load full cleaned data (no df.head(800)) ---
df = pd.read_csv("data/train_clean.csv")
print("Full dataset size:", len(df))
print(df.head())
print("Columns:", df.columns.tolist())

print("=> Splitting data...")
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["diagnosis"]
)
print(f"=> Split done: train {len(train_df)}, val {len(val_df)}")

train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
print("=> Splits saved!")


# --- Define transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --- Create datasets and loaders (num_workers=0 for Windows) ---
print("=> Creating dataset and loader...")
train_ds = DRDataset("data/train_split.csv", "data/train_images", transform)
val_ds = DRDataset("data/val_split.csv", "data/train_images", transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
print("=> Dataset and loader created!")


# --- Model setup: you can keep resnet18 (fast) or switch to resnet50 (final model) ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes: 0–4

# If you want the final, stronger model (slower but more accurate):
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# --- Training loop (with progress every 20 batches) ---
print("=> Starting training on FULL dataset...")
for epoch in range(3):  # you can increase this later (e.g., 10)
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1} / 3: iterating over train_loader...")

    for i, (images, labels) in enumerate(train_loader):
        if i == 0:
            print(f"  First batch loaded: images.size = {images.size()}, labels.size = {labels.size()}")

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print every 20 batches to see progress
        if (i + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}, batch {i+1}, running loss: {running_loss/(i+1):.4f}")

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


# --- Save final model ---
torch.save(model.state_dict(), "data/dr_model.pth")
print("Saved FINAL model to data/dr_model.pth")