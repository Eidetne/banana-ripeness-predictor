import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import os
import random

from utils import ripeness_to_days, predict_days, predict_random_image

# --------------------------
# Load trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

st.title("🍌 Banana Ripeness Predictor")
st.write("Upload a banana image or try a random one from the dataset!")

# Upload image
uploaded_file = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    pred_days = predict_days(model, image, device)
    st.success(f"Predicted: **{pred_days:.1f} days** to rotten")
else:
    if st.button("Try Random Dataset Image"):
        image, pred_days, true_days, class_name = predict_random_image(model, device)
        st.image(image, caption=f"Random Image ({class_name})", use_column_width=True)
        st.success(f"Predicted: **{pred_days:.1f} days** | True: {true_days:.1f} days")
