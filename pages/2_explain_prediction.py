
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from pytorch_grad_cam import GradCAMPlusPlus                       
from pytorch_grad_cam.utils.image import show_cam_on_image         
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform 

from src.utils import load_saved_model, predict_pil, TRANSFORM
from src.gradcam import generate_gradcam_overlay, get_target_layer     
from src.xai_metrics import (                                        
    iou_score, faithfulness_score,
    pointing_game_score, gradcam_vs_gradcampp_agreement
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

# Model config
MODEL_NAMES = ["resnet18", "densenet121", "vit_tiny"]                    
MODEL_LABELS = {
    "resnet18":    "ResNet-18",
    "densenet121": "DenseNet-121",
    "vit_tiny":    "ViT-Tiny"                                          
}

st.set_page_config(page_title="Explain Prediction", layout="wide")
st.title("🔬 Explain Prediction")
st.markdown("Upload a chest X-ray and see predictions + Grad-CAM heatmaps from all 3 models side by side.")

# Load models
@st.cache_resource
def load_all_models():
    return {m: load_saved_model(m) for m in MODEL_NAMES}

all_models = load_all_models()
available  = {k: v for k, v in all_models.items() if v is not None}

if not available:
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

# Upload
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
uploaded  = st.file_uploader("Upload a Chest X-Ray image", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("👆 Upload a chest X-ray to begin.")
    st.stop()

image   = Image.open(uploaded).convert("RGB")
orig_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
tensor  = TRANSFORM(image).unsqueeze(0)

st.divider()

#Section 1: Predictions
st.subheader("1. Predictions from All Models")
pred_cols = st.columns(len(available))
results   = {}

for col, (model_name, model) in zip(pred_cols, available.items()):
    prob, t, _ = predict_pil(model, image)
    pred = "PNEUMONIA 🔴" if prob >= threshold else "NORMAL 🟢"
    results[model_name] = {"prob": prob, "tensor": t, "pred": pred}
    with col:
        st.metric(MODEL_LABELS[model_name], pred, delta=f"{prob:.1%} pneumonia prob")

probs     = [results[m]["prob"] for m in available]
avg_prob  = float(np.mean(probs))
votes     = sum(1 for p in probs if p >= threshold)
consensus = "PNEUMONIA 🔴" if avg_prob >= threshold else "NORMAL 🟢"
st.info(f"**Ensemble:** {consensus} | Avg prob: {avg_prob:.1%} | Votes: {votes}/{len(available)}")
st.divider()

#Section 2: Grad-CAM heatmaps
st.subheader("2. Grad-CAM — Where did each model look?")
heat_cols = st.columns(len(available) + 1)
cam_store = {}  # save cams for later use in metrics

with heat_cols[0]:
    st.image(image.resize((224, 224)), caption="📷 Original", use_container_width=True)

for col, (model_name, model) in zip(heat_cols[1:], available.items()):
    with col:
        with st.spinner(f"{MODEL_LABELS[model_name]}..."):
            try:
                overlay, cam = generate_gradcam_overlay(
                    model, model_name,
                    results[model_name]["tensor"], orig_np
                )
                cam_store[model_name] = cam
                st.image(overlay, caption=f"🌡️ {MODEL_LABELS[model_name]}",
                         use_container_width=True)
                h, w = cam.shape
                quadrants = {
                    "Top-Left":     cam[:h//2, :w//2].mean(),
                    "Top-Right":    cam[:h//2, w//2:].mean(),
                    "Bottom-Left":  cam[h//2:, :w//2].mean(),
                    "Bottom-Right": cam[h//2:, w//2:].mean(),
                }
                top_quad = max(quadrants, key=quadrants.get)
                st.caption(f"📍 Focus: **{top_quad}**")
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

#Section 3: Auto Summary
st.divider()
st.subheader("3. Auto Summary")
focus_regions = []
for model_name in available:
    if model_name in cam_store:
        cam = cam_store[model_name]
        h, w = cam.shape
        quadrants = {
            "top-left lung":    cam[:h//2, :w//2].mean(),
            "top-right lung":   cam[:h//2, w//2:].mean(),
            "bottom-left lung": cam[h//2:, :w//2].mean(),
            "bottom-right lung":cam[h//2:, w//2:].mean(),
        }
        focus_regions.append(max(quadrants, key=quadrants.get))

if focus_regions:
    dominant = Counter(focus_regions).most_common(1)[0][0]
    agree    = len(set(focus_regions)) == 1
    if agree:
        summary = f"✅ All models consistently focused on the **{dominant}** region."
    else:
        summary = (f"⚠️ Models showed varied attention: **{', '.join(set(focus_regions))}**. "
                   f"Dominant focus: **{dominant}**.")
    st.markdown(f"> The ensemble predicts **{consensus}** with avg prob **{avg_prob:.1%}**.\n>\n> {summary}")

#Section 4: XAI Quality Metrics
st.divider()
st.subheader("4. XAI Quality Metrics")
st.markdown("Measures **how reliable** the heatmaps are — not just what they look like.")

metrics_rows = []
cam_plus_store = {}

for model_name, model in available.items():
    if model_name not in cam_store:
        continue
    try:
        cam = cam_store[model_name]

        # Grad-CAM++ heatmap
        reshape = vit_reshape_transform if model_name == "vit_tiny" else None
        with GradCAMPlusPlus(
            model=model,
            target_layers=get_target_layer(model, model_name),
            reshape_transform=reshape
        ) as cam_pp:
            cam_plus = cam_pp(
                input_tensor=results[model_name]["tensor"].to(device),
                targets=None
            )[0]
        cam_plus_store[model_name] = cam_plus

        concentration           = iou_score(cam)
        orig_p, mask_p, drop    = faithfulness_score(
            model, results[model_name]["tensor"], cam, device
        )
        agreement               = gradcam_vs_gradcampp_agreement(cam, cam_plus)
        top_point               = pointing_game_score(cam, top_n=1)[0]

        metrics_rows.append({
            "Model":                          MODEL_LABELS[model_name],
            "Concentration ↑":                f"{concentration:.3f}",
            "Faithfulness Drop ↑":            f"{drop:.3f}",
            "GradCAM / GradCAM++ Agreement ↑":f"{agreement:.3f}",
            "Top Activation (row, col)":      f"({top_point[0]}, {top_point[1]})"
        })
    except Exception as e:
        metrics_rows.append({"Model": MODEL_LABELS[model_name], "Error": str(e)})

if metrics_rows:
    st.dataframe(
        pd.DataFrame(metrics_rows).set_index("Model"),
        use_container_width=True
    )
    st.caption("""
    **Concentration ↑** Higher = model focused on a smaller, more precise region.
    **Faithfulness Drop ↑** Higher = masking the highlighted area drops confidence more → heatmap is meaningful.
    **Agreement ↑** Higher = Grad-CAM and Grad-CAM++ agree on the same region → stable explanation.
    """)

#Section 5: Grad-CAM vs Grad-CAM++
st.divider()
st.subheader("5. Grad-CAM vs Grad-CAM++")

for model_name, model in available.items():
    if model_name not in cam_store or model_name not in cam_plus_store:
        continue
    st.markdown(f"**{MODEL_LABELS[model_name]}**")
    c1, c2 = st.columns(2)

    overlay_base = show_cam_on_image(orig_np, cam_store[model_name], use_rgb=True)
    overlay_pp   = show_cam_on_image(orig_np, cam_plus_store[model_name], use_rgb=True)

    with c1:
        st.image(overlay_base, caption="Grad-CAM", use_container_width=True)
    with c2:
        st.image(overlay_pp,   caption="Grad-CAM++", use_container_width=True)
