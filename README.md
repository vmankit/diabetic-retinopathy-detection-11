# Diabetic Retinopathy Detection Web App

A Streamlit-based web application for automated detection of diabetic retinopathy using deep learning.

## Features

- **Image Upload**: Upload retinal fundus images (JPG, JPEG, PNG, BMP)
- **AI-Powered Analysis**: Uses a trained ResNet-50 model for classification
- **3-Class Classification**:
  - No DR (No Diabetic Retinopathy)
  - Non-Proliferative DR (Early stage)
  - Severe/Proliferative DR (Advanced stage)
- **Confidence Scores**: Displays prediction confidence with probability distribution
- **Visual Results**: Color-coded result boxes for easy interpretation

## Model Information

- **Architecture**: ResNet-50 (ImageNet V2 pretrained)
- **Dataset**: APTOS 2019 Blindness Detection
- **Classes**: 3-class system for diabetic retinopathy severity
- **Checkpoint**: `best_model.pth`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model file `best_model.pth` is in the project directory.

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Open the web application
2. Upload a retinal fundus image using the file uploader
3. Click on the uploaded image or wait for automatic analysis
4. View the prediction results with confidence scores
5. Check the probability distribution for detailed analysis

## Disclaimer

⚠️ **This tool is for educational and screening purposes only. It should not replace professional medical diagnosis. Always consult an ophthalmologist for accurate diagnosis and treatment.**

## Requirements

- Python 3.11
- PyTorch 2.2+
- Streamlit 1.33+
- PIL/Pillow
