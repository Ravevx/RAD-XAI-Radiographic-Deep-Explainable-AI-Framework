# pages/3_threshold_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import compute_metrics, get_roc_curve

st.set_page_config(page_title="Threshold Analysis", layout="wide")
st.title("📈 Threshold Analysis")
st.markdown("Adjust the decision threshold and see how sensitivity, specificity, and accuracy change in real time.")

MODEL_NAMES  = ["resnet18", "densenet121", "vit_tiny"]
MODEL_LABELS = {
    "resnet18":       "ResNet-18",
    "densenet121":    "DenseNet-121",
    "vit_tiny":    "ViT-Tiny" 
}

# Load scores
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
    st.error("No scores found. Run `python train.py` first.")
    st.stop()

# Controls
threshold = st.slider(
    "🎚️ Decision Threshold",
    min_value=0.01, max_value=0.99,
    value=0.50, step=0.01,
    help="Probability above this → PNEUMONIA"
)

st.divider()

#  Metrics table at current threshold 
st.subheader(f"Metrics at threshold = {threshold:.2f}")
rows = []
for model_name, df in scored_dfs.items():
    m = compute_metrics(df, threshold)
    m["Model"] = MODEL_LABELS[model_name]
    rows.append(m)

metrics_df = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    metrics_df[["AUC","Accuracy","Sensitivity","Specificity","TP","TN","FP","FN"]]
    .style.highlight_max(axis=0, subset=["Accuracy","Sensitivity","Specificity"], color="#d4edda")
    .highlight_min(axis=0, subset=["FP","FN"], color="#f8d7da")
    .format("{:.4f}", subset=["AUC","Accuracy","Sensitivity","Specificity"]),
    use_container_width=True
)

st.divider()

# Sensitivity/Specificity vs Threshold plot
st.subheader("Sensitivity & Specificity vs Threshold")
thresholds = np.arange(0.01, 1.0, 0.02)

fig, ax = plt.subplots(figsize=(10, 4))
for model_name, df in scored_dfs.items():
    sens_list = []
    spec_list = []
    for t in thresholds:
        m = compute_metrics(df, t)
        sens_list.append(m["Sensitivity"])
        spec_list.append(m["Specificity"])
    label = MODEL_LABELS[model_name]
    ax.plot(thresholds, sens_list, label=f"{label} Sensitivity", linestyle="-")
    ax.plot(thresholds, spec_list, label=f"{label} Specificity", linestyle="--")

ax.axvline(threshold, color="red", linestyle=":", linewidth=2, label=f"Current: {threshold:.2f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Sensitivity vs Specificity Trade-off")
ax.legend(loc="lower left", fontsize=7)
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close()

st.divider()

# ROC Curves
st.subheader("ROC Curves — All Models")
fig2, ax2 = plt.subplots(figsize=(7, 5))
for model_name, df in scored_dfs.items():
    fpr, tpr, _ = get_roc_curve(df)
    auc = compute_metrics(df)["AUC"]
    ax2.plot(fpr, tpr, label=f"{MODEL_LABELS[model_name]} (AUC={auc:.3f})")

ax2.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
ax2.set_xlabel("False Positive Rate (1 - Specificity)")
ax2.set_ylabel("True Positive Rate (Sensitivity)")
ax2.set_title("ROC Curves")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)
plt.close()

st.divider()

# Flip analysis 
st.subheader("Images That Flip at This Threshold")
st.markdown("Borderline cases — images close to the threshold that switch prediction as you move the slider.")

for model_name, df in scored_dfs.items():
    flip_margin = 0.08
    border_df = df[
        (df["prob"] >= threshold - flip_margin) &
        (df["prob"] <= threshold + flip_margin)
    ]
    st.write(f"**{MODEL_LABELS[model_name]}:** {len(border_df)} borderline images "
             f"(prob in [{threshold-flip_margin:.2f}, {threshold+flip_margin:.2f}])")
