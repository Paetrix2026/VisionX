import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/train.csv")

# Check what columns you actually have
print("Columns:", df.columns.tolist())

# Use the correct label column for stratification (e.g., "diagnosis")
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["diagnosis"]   # change "level" → "diagnosis"
)


train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)


print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Saved train_split.csv and val_split.csv")