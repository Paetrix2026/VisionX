import pandas as pd

df = pd.read_csv("data/train.csv")
print(df.head())
print(df.columns)

# Use the actual column name, e.g., "diagnosis"
print(df["diagnosis"].value_counts())