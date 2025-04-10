import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import gdown
import cv2
import numpy as np

# Import model classes
from Models.gan_model import Generator
from Models.unet_model import UNet
from Models.diffusion_model import SketchToColorDiffusionLite
from Models.traditional_model import FaceSketchColorizerLite
from Models.vit_model import TransformerUNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def download_weights(model_name):
    urls = {
        "GAN": "https://drive.google.com/uc?id=14o60RtT8LMeKR_8JMz8DXZlaAFH-opLU",
        "UNet": "https://drive.google.com/uc?id=1byMCLd9lybSZeblgDfXFAzs0DpQmcMjd",
        "Diffusion": "https://drive.google.com/uc?id=1FXlUhpMSlLk8IliizXE1HRfK_ArvHSzI",
        "ViT": "https://drive.google.com/uc?id=1h8VMOSQz7nM37ndG69eeYCsRrJ8d3UxN"
    }
    output = f"{model_name.lower()}_weights.pth"
    if not os.path.exists(output) and model_name in urls:
        gdown.download(urls[model_name], output, quiet=False)
    return output

@st.cache_resource
def load_model(model_name):
    if model_name == "Colorizer":
        return FaceSketchColorizerLite()

    weights_path = download_weights(model_name)

    if model_name == "GAN":
        model = Generator().to(device)
    elif model_name == "UNet":
        model = UNet().to(device)
    elif model_name == "ViT":
        model = TransformerUNet().to(device)  
    elif model_name == "Diffusion":
        model = SketchToColorDiffusionLite().to(device)  
    else:
        st.error("Invalid model selected.")
        st.stop()

    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
            st.stop()
    else:
        st.error("Model weights not found.")
        st.stop()

    model.eval()
    return model

def preprocess_image(image, model_name):
    if model_name == "UNet":
        image = np.array(image)
        image = cv2.resize(image, (178, 218))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(device)
    
    elif model_name == "ViT":
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(device)
    
    elif model_name == "Colorizer":
        image = image.convert("RGB")
        image = image.resize((256, 256))
        image = transforms.ToTensor()(image).unsqueeze(0)
        return image

    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        return transform(image).unsqueeze(0).to(device)


def postprocess_image(tensor, model_name):
    if model_name == "UNet":
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)
    else:
        tensor = tensor.squeeze(0)
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1).cpu().detach()
        return transforms.ToPILImage()(tensor)

# -------------------- UI --------------------
st.set_page_config(page_title="Sketch2Image Translator", layout="wide")
st.title("🎨 Sketch2Image Translator (Multi-Model Support)")

model_option = st.selectbox("Choose a model", ["GAN", "UNet", "ViT", "Diffusion", "Colorizer"])
model = load_model(model_option)

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")
        if model_option == "UNet":
            preview_image = input_image.resize((178, 218))
        elif model_option == "ViT":
            preview_image = input_image.resize((512, 512))
        else:
            preview_image = input_image.resize((256, 256))
        st.image(preview_image, caption="Uploaded Image", width=256)

        if st.button("Generate Output"):
            with st.spinner("Generating image..."):
                input_tensor = preprocess_image(input_image, model_option)

                with torch.no_grad():
                    if model_option == "Diffusion":
                        input_tensor = input_tensor.squeeze(0).to(device)
                        output_tensor = model(input_tensor, 1000)
                        output_image = postprocess_image(output_tensor, model_option)
                    elif model_option == "Colorizer":
                        output_image = model.predict(input_tensor)
                    else:
                        output_tensor = model(input_tensor)
                        output_image = postprocess_image(output_tensor, model_option)
                        output_image = output_image.resize((256, 256))
                col1, col2 = st.columns(2)
                with col1:
                    preview_image = preview_image.resize((256, 256))
                    st.image(preview_image, caption="Input")
                with col2:
                    st.image(output_image, caption="Output")

    except Exception as e:
        st.error(f"Error: {e}")
