# Diabetic Retinopathy Detection System

A deep learning-based web application for automated **diabetic retinopathy (DR)** screening from retinal fundus images. The system uses a PyTorch-based ResNet model to classify retinal scans into five clinical stages: **No DR, Mild, Moderate, Severe, and Proliferative DR**.

## Overview

Diabetic retinopathy is one of the leading causes of vision loss if not detected early. This project aims to support early-stage screening by combining image preprocessing, deep learning inference, and a lightweight Streamlit interface for interactive prediction.

The platform is designed for academic demonstration, hackathons, and prototype deployment. It provides an end-to-end workflow covering dataset preparation, model training, batch prediction, and web-based inference.

## Key Highlights

- Automated DR stage classification from fundus images.
- Five-class retinal disease grading pipeline.
- Custom training workflow built with PyTorch and torchvision.
- Validation split generation for model evaluation.
- Batch prediction support with CSV export.
- Interactive Streamlit frontend for real-time image upload and prediction.
- Modular project structure for easy experimentation and extension.

## Problem Statement

Manual screening of retinal fundus images can be time-consuming and requires expert ophthalmic review. In resource-constrained settings, delayed diagnosis can increase the risk of preventable blindness. This project addresses that gap by building a deep learning-based screening assistant that can quickly classify retinal images into diabetic retinopathy severity levels.

## Classification Labels

The model predicts one of the following five classes:

- **No DR**
- **Mild**
- **Moderate**
- **Severe**
- **Proliferative DR**

## System Architecture

The complete workflow of the project is:

1. **Data ingestion** from retinal image datasets and CSV metadata.
2. **Preprocessing and transformation** of fundus images.
3. **Train-validation split generation** for robust evaluation.
4. **Model training** using a ResNet-based deep learning architecture.
5. **Prediction on unseen images** using the saved trained model.
6. **Frontend result visualization** through a Streamlit web application.
7. **CSV export** for prediction records and downstream analysis.

## Tech Stack

### Programming Language

- Python

### Machine Learning / Deep Learning

- PyTorch
- torchvision
- NumPy
- scikit-learn
- Pillow

### Data Handling

- Pandas
- CSV-based metadata management

### Frontend

- Streamlit
- HTML/CSS styling within Streamlit UI

### Backend / Application Logic

- Python-based inference pipeline
- File handling for image upload and prediction
- Model loading and CPU inference using PyTorch

### Model

- ResNet-based image classification model
- Trained on retinal fundus images for 5-class diabetic retinopathy detection

## Software and Libraries Used

| Category         | Tools / Libraries                                   |
| ---------------- | --------------------------------------------------- |
| Language         | Python                                              |
| Deep Learning    | PyTorch, torchvision                                |
| Data Processing  | Pandas, NumPy                                       |
| ML Utilities     | scikit-learn                                        |
| Image Processing | Pillow                                              |
| Frontend         | Streamlit                                           |
| Backend Logic    | Python scripts (`train.py`, `predict.py`, `app.py`) |
| Model Storage    | `.pth` saved model weights                          |
| Output Format    | CSV                                                 |

## Project Structure

```text
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train_images/
│   └── test_images/
├── split_data.py
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset Structure

The dataset includes:

- `train.csv` — training image labels and metadata
- `test.csv` — test image list
- `sample_submission.csv` — prediction output reference
- `train_images/` — retinal fundus images for training
- `test_images/` — unseen images for inference

## Installation

Clone the repository and install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn pillow streamlit
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Split the dataset

```bash
python split_data.py
```

### 2. Train the model

```bash
python train.py
```

### 3. Run prediction on test data

```bash
python predict.py
```

### 4. Launch the web application

```bash
streamlit run app.py
```

## Output Files

After running the project, the following files are generated:

- `data/train_split.csv`
- `data/val_split.csv`
- `data/dr_model.pth`
- `data/test_predictions.csv`

## Frontend and Backend Description

### Frontend

The frontend is built using **Streamlit**, providing a lightweight and interactive interface for users to upload retinal images and view prediction results. It is suitable for demos, hackathons, and prototype healthcare AI applications.

### Backend

The backend is implemented in **Python** and handles:

- Loading the trained model
- Preprocessing uploaded images
- Running inference on CPU
- Mapping output logits to class labels
- Saving prediction results to CSV

## Model Details

- **Architecture:** ResNet-based CNN
- **Framework:** PyTorch
- **Task:** Multi-class image classification
- **Input:** Retinal fundus image
- **Output:** One of five diabetic retinopathy stages
- **Deployment Mode:** CPU inference via Streamlit app

## Use Cases

- Academic mini-projects and major projects
- Medical image classification demos
- Hackathon submissions in healthcare AI
- Prototype screening tools for diabetic retinopathy detection

## Future Enhancements

- Grad-CAM or saliency map explainability
- Better preprocessing for low-quality retinal images
- Class imbalance handling with weighted loss or augmentation
- Model comparison with EfficientNet / DenseNet / Vision Transformers
- Cloud or web deployment for public access
- Performance monitoring and confidence visualization

## Why This Project Stands Out

- Solves a real-world healthcare problem.
- Combines deep learning with an accessible web interface.
- Demonstrates both model development and practical deployment.
- Can be extended into a clinically assistive screening prototype.
