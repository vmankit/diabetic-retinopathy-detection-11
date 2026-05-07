import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

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
    'Non-Proliferative DR',
    'Severe / Proliferative DR'
]
CLASS_DESCRIPTIONS = {
    'No DR': 'No signs of diabetic retinopathy detected. Regular screening recommended.',
    'Non-Proliferative DR': 'Early diabetic retinopathy changes are present. Follow up with an ophthalmologist.',
    'Severe / Proliferative DR': 'Advanced diabetic retinopathy is present. Immediate medical attention is recommended.'
}

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_DIR / 'best_model.pth', APP_DIR / 'best_model (2).pth']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DRModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=512, dropout_rate=0.4):
        super(DRModel, self).__init__()

        base_model = models.resnet50(weights=None)

        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        self.avgpool = base_model.avgpool
        self.layer4 = base_model.layer4

        self.classifier = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

@st.cache_resource
def load_model():
    """Load the trained DRModel checkpoint"""
    try:
        model = DRModel(num_classes=3, hidden_dim=512, dropout_rate=0.4)

        model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"Missing checkpoint. Expected one of: {', '.join(path.name for path in MODEL_CANDIDATES)}"
            )

        state_dict = torch.load(model_path, map_location='cpu')

        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        if isinstance(state_dict, dict) and any(key.startswith('module.') for key in state_dict):
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        
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
    st.error("Failed to load the model. Please ensure 'best_model.pth' is in the current directory.")
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
            'Non-Proliferative DR': 'non-proliferative',
            'Severe / Proliferative DR': 'severe'
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
to classify retinal fundus images into three categories:

- **No DR**: No signs of diabetic retinopathy
- **Non-Proliferative DR**: Early diabetic retinopathy changes
- **Severe / Proliferative DR**: Advanced diabetic retinopathy changes

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
