import os
import pandas as pd
import shutil

csv_path = "data/train.csv"
image_dir = "data/train_images"
output_dir = "data_processed"

classes = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

# create folders
for c in classes.values():
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_name = row["id_code"] + ".png"
    label = row["diagnosis"]

    src = os.path.join(image_dir, img_name)
    dst = os.path.join(output_dir, classes[label], img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset prepared successfully!")