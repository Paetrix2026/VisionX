# Diabetic Retinopathy Detection System

## Project Overview

This project detects diabetic retinopathy from retinal fundus images using deep learning.  
It classifies retina images into 5 stages: No DR, Mild, Moderate, Severe, and Proliferative DR.

## Features

- Train a deep learning model on retinal images.
- Split data into training and validation sets.
- Predict diabetic retinopathy stages from test images.
- Simple frontend using Streamlit.
- Saves predictions to CSV format.

## Dataset

The dataset contains:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `train_images/`
- `test_images/`

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- scikit-learn
- torchvision
- Streamlit
- Pillow

## Project Structure

```text
project/
  data/
    train.csv
    test.csv
    sample_submission.csv
    train_images/
    test_images/
  split_data.py
  train.py
  predict.py
  app.py
  requirements.txt
```

## Installation

Install the required packages:

```bash
pip install torch torchvision pandas scikit-learn pillow streamlit
```

## How to Run

### 1. Split the data

```bash
python split_data.py
```

### 2. Train the model

```bash
python train.py
```

### 3. Predict on test images

```bash
python predict.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Output Files

- `data/train_split.csv`
- `data/val_split.csv`
- `data/dr_model.pth`
- `data/test_predictions.csv`

## Project Workflow

1. Load `train.csv`.
2. Split data into train and validation.
3. Train a ResNet50 model.
4. Save the trained model.
5. Use the model to predict test images.
6. Show results in the frontend.

## Future Scope

- Add Grad-CAM explainability.
- Improve preprocessing.
- Handle class imbalance better.
- Deploy the app online.

## Conclusion

This project helps detect diabetic retinopathy automatically from eye images and can support early screening.
