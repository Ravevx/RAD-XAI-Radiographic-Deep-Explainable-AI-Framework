import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve
)

#Config
MODELS     = ["resnet18", "densenet121", "vit_tiny"]
THRESHOLDS = [0.10, 0.50, 0.90]
FILTERS    = ["All", "Correct Only", "Misclassified Only"]

MODEL_LABELS = {
    "resnet18":    "ResNet-18",
    "densenet121": "DenseNet-121",
    "vit_tiny":    "ViT-Tiny"
}

#compute all metrics on a dataframe
def compute_metrics(df, threshold):
    """
    df must have columns: true_label (0/1), prob (float)
    Returns dict of metrics.
    """
    y_true = df["true_label"].values
    y_prob = df["prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    # AUC needs at least both classes present
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC":          round(float(auc), 4),
        "Accuracy":     round(float(acc), 4),
        "Sensitivity":  round(float(sensitivity), 4),
        "Specificity":  round(float(specificity), 4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }

#Main evaluation loop
full_report_rows      = []
threshold_report_rows = []

for model_name in MODELS:
    csv_path = f"models/{model_name}_test_scores.csv"

    if not os.path.exists(csv_path):
        print(f"Skipping {model_name} — {csv_path} not found.")
        continue

    df_all = pd.read_csv(csv_path)

    # Validate required columns
    required = {"true_label", "prob"}
    if not required.issubset(df_all.columns):
        print(f"{csv_path} missing columns: {required - set(df_all.columns)}")
        continue

    print(f"\n{'='*55}")
    print(f"Model: {MODEL_LABELS[model_name]}  |  Total test images: {len(df_all)}")

    #Part 1: Full report (all thresholds × all filters)
    for threshold in THRESHOLDS:
        df_all["pred_label"] = (df_all["prob"] >= threshold).astype(int)
        df_all["correct"]    = df_all["pred_label"] == df_all["true_label"]

        subsets = {
            "All":                df_all,
            "Correct Only":       df_all[df_all["correct"]],
            "Misclassified Only": df_all[~df_all["correct"]],
        }

        for filter_name, df_sub in subsets.items():
            total = len(df_sub)

            if total == 0:
                row = {
                    "Model":        MODEL_LABELS[model_name],
                    "Threshold":    threshold,
                    "Filter":       filter_name,
                    "Total Images": 0,
                    "AUC":          "N/A",
                    "Accuracy":     "N/A",
                    "Sensitivity":  "N/A",
                    "Specificity":  "N/A",
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                }
            else:
                try:
                    metrics = compute_metrics(df_sub, threshold)
                except Exception as e:
                    metrics = {
                        "AUC": "ERR", "Accuracy": "ERR",
                        "Sensitivity": "ERR", "Specificity": "ERR",
                        "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                        "_error": str(e)
                    }

                row = {
                    "Model":        MODEL_LABELS[model_name],
                    "Threshold":    threshold,
                    "Filter":       filter_name,
                    "Total Images": total,
                    **metrics
                }

            full_report_rows.append(row)
            print(f"  thresh={threshold} | {filter_name:20s} | "
                  f"n={total:4d} | "
                  f"AUC={row.get('AUC', 'N/A'):6} | "
                  f"Acc={row.get('Accuracy', 'N/A'):6} | "
                  f"Sens={row.get('Sensitivity', 'N/A'):6} | "
                  f"Spec={row.get('Specificity', 'N/A'):6}")

    #Part 2: Threshold analysis (fine-grained, All filter only)
    fine_thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    for t in fine_thresholds:
        try:
            m = compute_metrics(df_all, t)
            threshold_report_rows.append({
                "Model":       MODEL_LABELS[model_name],
                "Threshold":   float(t),
                "Sensitivity": m["Sensitivity"],
                "Specificity": m["Specificity"],
                "Accuracy":    m["Accuracy"],
                "AUC":         m["AUC"],
                "TP": m["TP"], "TN": m["TN"],
                "FP": m["FP"], "FN": m["FN"],
            })
        except Exception:
            pass

#Save outputs
os.makedirs("models", exist_ok=True)

#Full evaluation report
full_df = pd.DataFrame(full_report_rows)
full_df = full_df[[
    "Model", "Threshold", "Filter", "Total Images",
    "AUC", "Accuracy", "Sensitivity", "Specificity",
    "TP", "TN", "FP", "FN"
]]
full_df.to_csv("models/full_evaluation_report.csv", index=False)
print(f"\nSaved: models/full_evaluation_report.csv  ({len(full_df)} rows)")

# Threshold analysis
thresh_df = pd.DataFrame(threshold_report_rows)
thresh_df.to_csv("models/threshold_analysis.csv", index=False)
print(f"Saved: models/threshold_analysis.csv  ({len(thresh_df)} rows)")

#Pretty print the 3-threshold summary
print(f"\n{'='*55}")
print("SUMMARY — All models at threshold 0.50 (All images)")
print(f"{'='*55}")
summary = full_df[
    (full_df["Threshold"] == 0.50) &
    (full_df["Filter"]    == "All")
][["Model", "Total Images", "AUC", "Accuracy", "Sensitivity", "Specificity"]]
print(summary.to_string(index=False))
print(f"{'='*55}")
