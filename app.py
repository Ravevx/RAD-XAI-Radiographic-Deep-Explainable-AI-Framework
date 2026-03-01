# app.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="PneumoXAI — Chest X-Ray Explainability Suite",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 PneumoXAI — Chest X-Ray Explainability Suite")
st.markdown("""
A clinician-grade AI tool for pneumonia detection and explainability.
Combines **3 deep learning backbones** with **Grad-CAM** to show *why* each model makes its decision.
""")

st.divider()

#Model status
st.subheader("Model Status")
col1, col2, col3 = st.columns(3)
models_info = {
    "ResNet-18":    "models/resnet18.pth",
    "DenseNet-121": "models/densenet121.pth",
    "ViT-Tiny":     "models/vit_tiny.pth", 
}

for col, (name, path) in zip([col1, col2, col3], models_info.items()):
    with col:
        ready = os.path.exists(path)
        st.metric(
            label=name,
            value="✅ Ready" if ready else "❌ Not trained",
        )

# Metrics summary
st.divider()
st.subheader("Performance Summary")
metrics_path = "models/all_metrics.csv"
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path, index_col=0)
    float_cols = [c for c in ["AUC","Accuracy","Sensitivity","Specificity"] if c in df.columns]

    st.dataframe(
        df.style
          .highlight_max(axis=0, subset=float_cols, color="#d4edda")
          .format("{:.4f}", subset=float_cols),
        use_container_width=True
    )
else:
    st.warning("⚠️ Run `python train.py` first to train all models.")

st.divider()

#Navigation guide
st.subheader("Pages")
c1, c2, c3 = st.columns(3)
with c1:
    st.info("**📋 Page 1 — Patient Browser**\nBrowse all test X-rays. Filter misclassifications per model.")
with c2:
    st.info("**🔬 Page 2 — Explain Prediction**\nUpload any X-ray. Get predictions + Grad-CAM heatmaps from all 3 models side by side.")
with c3:
    st.info("**📈 Page 3 — Threshold Analysis**\nAdjust decision threshold. See sensitivity, specificity, and ROC curve update live.")

st.caption("Use the sidebar ← to navigate between pages.")
