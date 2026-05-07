import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .no-dr {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .non-proliferative {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .mild-npdr {
        background-color: #e2f0fb;
        border: 1px solid #b8d7f0;
        color: #0c5460;
    }
    .moderate-npdr {
        background-color: #fce8d5;
        border: 1px solid #f6c08b;
        color: #8a4b08;
    }
    .severe {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .proliferative-dr {
        background-color: #e6d4f5;
        border: 1px solid #c7a5e6;
        color: #4b2a63;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div class="main-header">
    <h1>👁️ Diabetic Retinopathy Detection</h1>
    <p>Upload a retinal fundus image to detect diabetic retinopathy using deep learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Class labels
CLASS_NAMES = [
    'No DR',
    'Mild NPDR',
    'Moderate NPDR',
    'Severe NPDR',
    'Proliferative DR'
]
CLASS_DESCRIPTIONS = {
    'No DR': 'No signs of diabetic retinopathy detected. Regular screening recommended.',
    'Mild NPDR': 'Very early diabetic retinopathy changes are present. Follow up with an ophthalmologist.',
    'Moderate NPDR': 'Moderate diabetic retinopathy changes are present. Clinical follow-up is recommended.',
    'Severe NPDR': 'Severe non-proliferative diabetic retinopathy is present. Prompt specialist review is recommended.',
    'Proliferative DR': 'Advanced proliferative diabetic retinopathy is present. Immediate medical attention is recommended.'
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DRModel(nn.Module):
    def __init__(self):
        super(DRModel, self).__init__()

        base_model = models.resnet50(weights=None)

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    """Load the trained DRModel checkpoint"""
    try:
        model = DRModel()

        model_path = 'best_model (2).pth'
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        if isinstance(state_dict, dict) and any(key.startswith('module.') for key in state_dict):
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image):
    """Make prediction on the uploaded image"""
    try:
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_index = int(predicted.item())
        
        return CLASS_NAMES[predicted_index], confidence.item(), probabilities[0].tolist()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0, [0, 0, 0]

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please ensure 'best_model (2).pth' is in the current directory.")
    st.stop()

# File upload
st.subheader("Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose a retinal fundus image...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload a clear retinal fundus image for analysis"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image")
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        prediction, confidence, probabilities = predict_image(model, image)
    
    if prediction:
        # Display results
        st.subheader("Analysis Results")
        
        # Determine CSS class based on prediction
        css_class_map = {
            'No DR': 'no-dr',
            'Mild NPDR': 'non-proliferative',
            'Moderate NPDR': 'moderate-npdr',
            'Severe NPDR': 'severe',
            'Proliferative DR': 'proliferative-dr'
        }
        css_class = css_class_map.get(prediction, 'severe')
        
        st.markdown(f"""
        <div class="result-box {css_class}">
            <h3>Prediction: {prediction}</h3>
            <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
            <p><strong>Description:</strong> {CLASS_DESCRIPTIONS[prediction]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display probability distribution
        st.subheader("Probability Distribution")
        prob_data = {
            'Class': CLASS_NAMES,
            'Probability': [p * 100 for p in probabilities]
        }
        st.bar_chart(prob_data, x='Class', y='Probability')
        
        # Display detailed probabilities
        with col2:
            st.markdown("### Detailed Probabilities")
            for class_name, prob in zip(CLASS_NAMES, probabilities):
                st.metric(
                    label=class_name,
                    value=f"{prob * 100:.2f}%",
                    delta=""
                )

# Information section
st.markdown("---")
st.markdown("""
### About This Application

This application uses a deep learning model based on ResNet-50 trained on the APTOS 2019 Blindness Detection dataset 
to classify retinal fundus images into five categories:

- **No DR**: No signs of diabetic retinopathy
- **Mild NPDR**: Mild non-proliferative diabetic retinopathy
- **Moderate NPDR**: Moderate non-proliferative diabetic retinopathy
- **Severe NPDR**: Severe non-proliferative diabetic retinopathy
- **Proliferative DR**: Proliferative diabetic retinopathy

⚠️ **Disclaimer**: This tool is for educational and screening purposes only. It should not replace 
professional medical diagnosis. Always consult an ophthalmologist for accurate diagnosis and treatment.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Built with Streamlit & PyTorch | Diabetic Retinopathy Detection</p>
</div>
""", unsafe_allow_html=True)
