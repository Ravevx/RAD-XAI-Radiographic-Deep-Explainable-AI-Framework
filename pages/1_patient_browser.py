# pages/1_patient_browser.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import load_saved_model, compute_metrics

st.set_page_config(page_title="Patient Browser", layout="wide")
st.title("📋 Patient Browser")
st.markdown("Browse all test X-rays, see ground-truth vs model predictions, filter misclassifications.")

MODEL_NAMES = ["resnet18", "densenet121", "vit_tiny"]
MODEL_LABELS = {
    "resnet18":"ResNet-18",
    "densenet121":"DenseNet-121",
    "vit_tiny": "ViT-Tiny"
}

#Load scored CSVs
@st.cache_data
def load_all_scores():
    dfs = {}
    for m in MODEL_NAMES:
        path = f"models/{m}_test_scores.csv"
        if os.path.exists(path):
            dfs[m] = pd.read_csv(path)
    return dfs

scored_dfs = load_all_scores()

if not scored_dfs:
    st.error("No model scores found. Run `python train.py` first.")
    st.stop()

#Sidebar controls
st.sidebar.header("Filters")
selected_model = st.sidebar.selectbox(
    "Model", MODEL_NAMES,
    format_func=lambda x: MODEL_LABELS[x]
)
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

show_filter = st.sidebar.radio(
    "Show",
    ["All", "Correct Only", "Misclassified Only"]
)

df = scored_dfs[selected_model].copy()
df["pred_label"] = (df["prob"] >= threshold).astype(int)
df["pred_name"]  = df["pred_label"].map({0:"NORMAL", 1:"PNEUMONIA"})
df["correct"]    = df["pred_label"] == df["true_label"]
df["status"]     = df["correct"].map({True:"✅ Correct", False:"❌ Wrong"})

if show_filter == "Correct Only":
    df = df[df["correct"]]
elif show_filter == "Misclassified Only":
    df = df[~df["correct"]]

#Metrics strip
metrics = compute_metrics(df, threshold)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("AUC",         f"{metrics['AUC']:.3f}")
col2.metric("Accuracy",    f"{metrics['Accuracy']:.1%}")
col3.metric("Sensitivity", f"{metrics['Sensitivity']:.1%}")
col4.metric("Specificity", f"{metrics['Specificity']:.1%}")
col5.metric("Showing",     f"{len(df)} images")

st.divider()

#Image grid
st.subheader(f"{MODEL_LABELS[selected_model]} — Test Set")
N_COLS = 4
rows = [df.iloc[i:i+N_COLS] for i in range(0, min(len(df), 40), N_COLS)]

for row_df in rows:
    cols = st.columns(N_COLS)
    for col, (_, row) in zip(cols, row_df.iterrows()):
        with col:
            try:
                img = Image.open(row["path"]).convert("RGB")
                st.image(img, use_container_width=True)
                color = "🟢" if row["correct"] else "🔴"
                st.caption(
                    f"{color} True: **{row['true_name']}**\n"
                    f"Pred: **{row['pred_name']}** ({row['prob']:.0%})"
                )
            except Exception:
                st.caption("Image not found")
