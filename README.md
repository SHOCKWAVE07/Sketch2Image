# Sketch2Real 🎨📷
**AI-Driven Sketch-to-Photo Transformation**

Welcome to **Sketch2Real**, an end-to-end deep learning pipeline that transforms black-and-white sketches into realistic color photos using cutting-edge generative models. Our project explores both traditional and modern deep learning techniques to solve the sketch-to-image translation problem.

---

## 🚀 Project Highlights

- 🔍 **Objective**: Convert black-and-white face sketches into colored, photo-realistic images.
- 🧠 **Techniques Used**:
  - Traditional image processing for baseline comparison
  - Deep learning models: U-Net, GAN, Vision Transformer (ViT), and Diffusion Model
  - Rule-based colorization using traditional heuristics

---

## 🧩 Model Architectures

### ✅ 1. U-Net (CNN)
- Encoder-Decoder with skip connections
- Optimized for fine-grained facial features

### 🎭 2. GAN
- Generator learns to translate sketches
- Discriminator distinguishes real vs. generated photos

### 🧠 3. Vision Transformer (ViT)
- Patch-based attention mechanism
- Transformer encoder blocks for contextual image generation

### 🌫️ 4. Diffusion Model
- Iterative denoising from noise to high-fidelity image
- Uses U-Net scheduler in reverse-time diffusion

### 🧪 5. Traditional Rule-Based Method
- `FaceSketchColorizer`: A lightweight rule-based method using skin detection and heuristics for rough colorization

---

## 📱 Streamlit App

We created a simple and modular front-end using **Streamlit**, allowing users to:

- Upload a sketch
- Choose a model (U-Net, GAN, ViT, Diffusion, Traditional)
- View the output in real-time

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sketch2real.git
cd sketch2real
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```
