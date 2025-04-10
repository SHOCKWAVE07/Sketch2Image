import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Import all model classes
from gan_model import Generator  # GAN
# from vit_model import TransformerUNet     
# from diffusion_model import DiffusionPredictor  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    if model_name == "GAN":
        model = Generator().to(device)
        weights_path = "generator.pth"
    elif model_name == "ViT":
        model = Generator().to(device)
        weights_path = "generator.pth"
    elif model_name == "Diffusion":
        model = Generator().to(device)
        weights_path = "generator.pth"
    else:
        st.error("Invalid model selected.")
        st.stop()

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.error(f"Model weights not found for {model_name}!")
        st.stop()

    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1).cpu().detach()
    return transforms.ToPILImage()(tensor)

# --- Streamlit UI ---
st.title("Multi-Model Image Translator")

# Select model
model_option = st.selectbox("Select model", ["GAN", "ViT", "Diffusion"])
model = load_model(model_option)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Input Image", use_container_width=True)

        if st.button("Generate Output"):
            with st.spinner("Generating..."):
                input_tensor = preprocess_image(input_image)

                with torch.no_grad():
                    if model_option == "Diffusion":
                        output_tensor = model.predict(input_tensor)  # Assuming Diffusion needs a special call
                    else:
                        output_tensor = model(input_tensor)

                output_image = postprocess_image(output_tensor)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(input_image.resize((256, 256)), caption="Input Image")
                with col2:
                    st.image(output_image.resize((256, 256)), caption="Output Image")
    except Exception as e:
        st.error(f"Error: {e}")
