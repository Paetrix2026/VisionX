import os
import pandas as pd

df = pd.read_csv("data/train.csv")
img_dir = "data/train_images"

def exists_with_ext(code, exts=(".jpeg", ".jpg", ".png")):
    for ext in exts:
        path = os.path.join(img_dir, f"{code}{ext}")
        if os.path.exists(path):
            return path
    return None

df["img_path"] = df["id_code"].apply(lambda c: exists_with_ext(c))
df = df[df["img_path"].notna()].copy()
df.to_csv("data/train_clean.csv", index=False)
print(f"Cleaned dataset size: {len(df)}")