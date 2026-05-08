import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Result boxes */
    .result-box {
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
    }
    .no-dr        { background:#d1fae5; border:1px solid #6ee7b7; color:#065f46; }
    .mild-npdr    { background:#dbeafe; border:1px solid #93c5fd; color:#1e40af; }
    .moderate-npdr{ background:#fef9c3; border:1px solid #fde047; color:#854d0e; }
    .severe-npdr  { background:#ffedd5; border:1px solid #fb923c; color:#9a3412; }
    .pdr          { background:#fee2e2; border:1px solid #f87171; color:#7f1d1d; }

    /* DR Scale bar */
    .scale-container {
        margin: 1.5rem 0;
    }
    .scale-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.6rem;
    }
    .scale-bar {
        display: flex;
        border-radius: 12px;
        overflow: hidden;
        height: 46px;
    }
    .scale-segment {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 600;
        color: white;
        cursor: default;
        transition: filter 0.2s;
        padding: 0 2px;
        text-align: center;
        line-height: 1.2;
    }
    .scale-segment.active {
        outline: 3px solid #1e293b;
        outline-offset: -3px;
        filter: brightness(1.08);
        z-index: 2;
        position: relative;
    }
    .scale-segment.inactive {
        filter: opacity(0.38) grayscale(0.4);
    }
    .seg-0 { background: #22c55e; }
    .seg-1 { background: #84cc16; }
    .seg-2 { background: #eab308; }
    .seg-3 { background: #f97316; }
    .seg-4 { background: #ef4444; }

    .grade-badge {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        width: 64px;
        height: 64px;
        line-height: 64px;
        text-align: center;
        border-radius: 50%;
        color: white;
        margin-bottom: 0.5rem;
    }
    .badge-0 { background: #22c55e; }
    .badge-1 { background: #84cc16; }
    .badge-2 { background: #eab308; color: #422006; }
    .badge-3 { background: #f97316; }
    .badge-4 { background: #ef4444; }

    .result-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-title { margin: 0; font-size: 1.2rem; font-weight: 700; }
    .result-sub   { margin: 0.2rem 0 0; font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>👁️ Diabetic Retinopathy Detection</h1>
    <p>Upload a retinal fundus image to detect DR severity using deep learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── DR Grading Scale (0-4) ─────────────────────────────────────────────────
# The model outputs 3 classes which map to grade groups:
#   Class 0 → Grade 0        (No DR)
#   Class 1 → Grades 1 & 2   (Mild / Moderate NPDR)
#   Class 2 → Grades 3 & 4   (Severe NPDR / PDR)
#
# For a richer UX we show the full 0-4 scale and pick the most likely grade
# within the predicted group based on confidence.

DR_GRADE_INFO = {
    0: {
        "name": "No DR",
        "abbr": "Grade 0",
        "css_class": "no-dr",
        "badge_class": "badge-0",
        "seg_class": "seg-0",
        "description": "No signs of diabetic retinopathy. The retina appears healthy. "
                       "Continue regular annual eye examinations.",
        "recommendations": [
            "Maintain good blood sugar control",
            "Annual eye exam recommended",
            "Monitor HbA1c levels",
        ],
    },
    1: {
        "name": "Mild NPDR",
        "abbr": "Grade 1",
        "css_class": "mild-npdr",
        "badge_class": "badge-1",
        "seg_class": "seg-1",
        "description": "Mild Non-Proliferative Diabetic Retinopathy. A few microaneurysms "
                       "are present. Follow up with an ophthalmologist within 12 months.",
        "recommendations": [
            "Ophthalmology review in 12 months",
            "Optimise blood glucose & blood pressure",
            "Lifestyle modifications advised",
        ],
    },
    2: {
        "name": "Moderate NPDR",
        "abbr": "Grade 2",
        "css_class": "moderate-npdr",
        "badge_class": "badge-2",
        "seg_class": "seg-2",
        "description": "Moderate Non-Proliferative Diabetic Retinopathy. More pronounced "
                       "vascular changes detected. Review by an ophthalmologist within 6 months.",
        "recommendations": [
            "Ophthalmology review within 6 months",
            "Strict glycaemic control required",
            "Check kidney function & blood pressure",
        ],
    },
    3: {
        "name": "Severe NPDR",
        "abbr": "Grade 3",
        "css_class": "severe-npdr",
        "badge_class": "badge-3",
        "seg_class": "seg-3",
        "description": "Severe Non-Proliferative Diabetic Retinopathy. Extensive retinal "
                       "changes present. Urgent referral to an ophthalmologist is recommended.",
        "recommendations": [
            "Urgent ophthalmology referral (within weeks)",
            "Consider retinal laser treatment",
            "Maximise systemic risk factor control",
        ],
    },
    4: {
        "name": "Proliferative DR",
        "abbr": "Grade 4",
        "css_class": "pdr",
        "badge_class": "badge-4",
        "seg_class": "seg-4",
        "description": "Proliferative Diabetic Retinopathy. New fragile blood vessels are "
                       "growing on the retina. Immediate specialist intervention is required.",
        "recommendations": [
            "Immediate ophthalmology intervention",
            "Anti-VEGF injections or laser photocoagulation",
            "Risk of severe vision loss — act urgently",
        ],
    },
}

# Model class → DR grade range mapping
CLASS_TO_GRADE_RANGE = {
    0: [0],        # No DR
    1: [1, 2],     # Mild / Moderate NPDR
    2: [3, 4],     # Severe NPDR / PDR
}

CLASS_NAMES = ['No DR', 'Non-Proliferative DR', 'Severe / Proliferative DR']

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_DIR / 'best_model.pth', APP_DIR / 'best_model (2).pth']

# Image preprocessing
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def remove_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 10 or h < 10:
        return image
    return image[y:y + h, x:x + w]


def resize_image(image, size=IMAGE_SIZE):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def ben_graham_filter(image, sigma=10):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image, 4, blurred, -4, 128)


def preprocess_image(image, size=IMAGE_SIZE):
    image = remove_black_borders(image)
    image = resize_image(image, size)
    image = apply_clahe(image)
    image = ben_graham_filter(image)
    return image


def image_to_tensor(image):
    image_array = np.array(image) if isinstance(image, Image.Image) else image
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    processed_bgr = preprocess_image(image_array)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(processed_rgb)


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
        self.avgpool  = base_model.avgpool
        self.layer4   = base_model.layer4

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

        model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"Missing checkpoint. Expected one of: {', '.join(p.name for p in MODEL_CANDIDATES)}"
            )

        state_dict = torch.load(model_path, map_location='cpu')

        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def predict_image(model, image):
    """Make prediction and map to DR 0-4 grade."""
    try:
        image_tensor = image_to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            outputs      = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_index = int(predicted.item())

        probs_list = probabilities[0].tolist()
        grade_range = CLASS_TO_GRADE_RANGE[predicted_index]

        # Pick sub-grade within range: high confidence → upper end of range
        if len(grade_range) == 1:
            dr_grade = grade_range[0]
        else:
            # confidence > 0.75 → more severe sub-grade, else milder
            dr_grade = grade_range[1] if confidence.item() > 0.75 else grade_range[0]

        return predicted_index, dr_grade, confidence.item(), probs_list
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, 0, [0, 0, 0]


def render_scale_bar(active_grade: int):
    segments_html = ""
    for g in range(5):
        if g == active_grade:
            state = "active"
        else:
            state = "inactive"
        label = DR_GRADE_INFO[g]["abbr"]
        short = DR_GRADE_INFO[g]["name"].replace(" NPDR", "").replace("Proliferative DR", "PDR")
        segments_html += (
            f'<div class="scale-segment seg-{g} {state}">'
            f'  <span>{label}</span>'
            f'  <span style="font-size:0.62rem;opacity:0.9">{short}</span>'
            f'</div>'
        )
    return f"""
    <div class="scale-container">
        <div class="scale-label">DR Severity Scale (Grade 0 – 4)</div>
        <div class="scale-bar">{segments_html}</div>
    </div>
    """


# ─── Load model ─────────────────────────────────────────────────────────────
model = load_model()

if model is None:
    st.error("Failed to load the model. Please ensure 'best_model.pth' is in the current directory.")
    st.stop()

# ─── File upload ─────────────────────────────────────────────────────────────
st.subheader("Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose a retinal fundus image...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload a clear retinal fundus image for analysis"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analysing image…"):
        class_idx, dr_grade, confidence, probabilities = predict_image(model, image)

    if dr_grade is not None:
        info = DR_GRADE_INFO[dr_grade]

        # ── Grade badge + result box ────────────────────────────────────────
        st.subheader("Analysis Results")

        st.markdown(f"""
        <div class="result-box {info['css_class']}">
            <div class="result-header">
                <div class="grade-badge {info['badge_class']}">{dr_grade}</div>
                <div>
                    <p class="result-title">{info['abbr']} — {info['name']}</p>
                    <p class="result-sub"><strong>Confidence:</strong> {confidence * 100:.1f}%</p>
                </div>
            </div>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── DR Scale bar ────────────────────────────────────────────────────
        st.markdown(render_scale_bar(dr_grade), unsafe_allow_html=True)

        # ── Recommendations ─────────────────────────────────────────────────
        st.markdown("#### Clinical Recommendations")
        for rec in info["recommendations"]:
            st.markdown(f"- {rec}")

        # ── Probability distribution ────────────────────────────────────────
        st.subheader("Model Class Probabilities")
        prob_data = {
            'Class': CLASS_NAMES,
            'Probability (%)': [p * 100 for p in probabilities]
        }
        st.bar_chart(prob_data, x='Class', y='Probability (%)')

        # ── Detailed probabilities in second column ─────────────────────────
        with col2:
            st.markdown("### Class Breakdown")
            for class_name, prob in zip(CLASS_NAMES, probabilities):
                st.metric(label=class_name, value=f"{prob * 100:.2f}%")

# ─── About section ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### DR Grading Scale (0 – 4)

| Grade | Name | Key Features |
|:---:|---|---|
| **0** | No DR | No abnormalities |
| **1** | Mild NPDR | Microaneurysms only |
| **2** | Moderate NPDR | More than just microaneurysms, less than severe |
| **3** | Severe NPDR | 20+ intraretinal haemorrhages in 4 quadrants |
| **4** | Proliferative DR | Neovascularisation / vitreous haemorrhage |

---

### About This Application

This application uses a deep learning model based on **ResNet-50** trained on the APTOS 2019 Blindness Detection dataset.  
The model predicts one of three broad DR classes which are then mapped to the standard **International Clinical DR Scale (0–4)**.

⚠️ **Disclaimer**: This tool is for educational and screening purposes only. It should **not** replace professional medical diagnosis. Always consult an ophthalmologist for accurate diagnosis and treatment.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85rem;">
    <p>Built with Streamlit &amp; PyTorch | Diabetic Retinopathy Detection | DR Grade Scale 0–4</p>
</div>
""", unsafe_allow_html=True)
