# xai_quality_analysis.py
# Run: python xai_quality_analysis.py
# Output: models/xai_quality_full.csv
#
# Runs ALL 3 models on EVERY test image.
# For each image × model: computes Grad-CAM, Grad-CAM++,
# Concentration, Faithfulness Drop, Agreement, Top Activation.

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

sys.path.append(os.path.dirname(__file__))
from src.utils import load_saved_model, TRANSFORM

#Config
TEST_DIR   = "data/chest_xray/test"
OUTPUT_CSV = "models/xai_quality_full.csv"
IMG_SIZE   = 224
HALF       = IMG_SIZE // 2

MODELS = ["resnet18", "densenet121", "vit_tiny"]
MODEL_LABELS = {
    "resnet18":    "ResNet-18",
    "densenet121": "DenseNet-121",
    "vit_tiny":    "ViT-Tiny"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Target layer mapping 
def get_target_layer(model, model_name):
    if model_name == "resnet18":
        return [model.layer4[-1]]
    elif model_name == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    elif model_name == "vit_tiny":
        return [model.blocks[-1].norm1]
    else:
        raise ValueError(f"Unknown: {model_name}")

def get_reshape(model_name):
    return vit_reshape_transform if model_name == "vit_tiny" else None

#  XAI metric functions 
def concentration(cam, threshold=0.5):
    binary = (cam > threshold).astype(float)
    activated = binary.sum()
    total = cam.shape[0] * cam.shape[1]
    return round(float(1.0 - activated / total), 4)

def faithfulness_drop(model, tensor, cam, target_class):
    model.eval()
    t = tensor.to(device)

    with torch.no_grad():
        orig_prob = torch.softmax(model(t), dim=1)[0][target_class].item()

    thresh = np.percentile(cam, 50)
    mask = torch.tensor(
        (cam < thresh).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device).expand_as(t)

    with torch.no_grad():
        masked_prob = torch.softmax(model(t * mask), dim=1)[0][target_class].item()

    return round(float(orig_prob - masked_prob), 4)

def agreement(cam1, cam2, threshold=0.5):
    b1 = (cam1 > threshold).astype(float)
    b2 = (cam2 > threshold).astype(float)
    intersection = (b1 * b2).sum()
    union = ((b1 + b2) > 0).sum()
    return round(float(intersection / union) if union > 0 else 0.0, 4)

def top_activation(cam):
    idx = np.argmax(cam.ravel())
    row, col = np.unravel_index(idx, cam.shape)
    return int(row), int(col)

def quadrant(row, col):
    v = "Top"    if row < HALF else "Bottom"
    h = "Left"   if col < HALF else "Right"
    return f"{v}-{h}"

# Load all models once 
print("\nLoading models...")
loaded_models = {}
for model_name in MODELS:
    m = load_saved_model(model_name)
    if m is None:
        print(f" {model_name} not found — skipping")
        continue
    m.eval().to(device)
    loaded_models[model_name] = m
    print(f" {MODEL_LABELS[model_name]} loaded")

if not loaded_models:
    print("No models found. Run python train.py first.")
    sys.exit(1)

#Collect all test images
print("\nScanning test directory...")
image_records = []
for label_name, label_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
    folder = os.path.join(TEST_DIR, label_name)
    if not os.path.exists(folder):
        print(f" Folder not found: {folder}")
        continue
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_records.append({
                "filename":   fname,
                "path":       os.path.join(folder, fname),
                "true_label": label_idx,
                "true_name":  label_name
            })

print(f"  Found {len(image_records)} test images")

#Main loop
rows = []
total = len(image_records) * len(loaded_models)
done  = 0
start = time.time()

print(f"\nRunning XAI analysis on {len(image_records)} images × {len(loaded_models)} models = {total} total...\n")

for img_rec in image_records:
    # Load image once per image (shared across models)
    try:
        pil_img = Image.open(img_rec["path"]).convert("RGB")
    except Exception as e:
        print(f"Could not open {img_rec['filename']}: {e}")
        continue

    orig_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(device)

    for model_name, model in loaded_models.items():
        done += 1
        try:
            # Forward pass (prediction) 
            with torch.no_grad():
                out   = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred_class = probs.argmax().item()
                pred_prob  = probs[1].item()

            target_layers = get_target_layer(model, model_name)
            reshape       = get_reshape(model_name)

            # rad-CAM
            with GradCAM(
                model=model,
                target_layers=target_layers,
                reshape_transform=reshape
            ) as gc:
                cam = gc(input_tensor=tensor, targets=None)[0]

            #Grad-CAM++ 
            with GradCAMPlusPlus(
                model=model,
                target_layers=target_layers,
                reshape_transform=reshape
            ) as gcpp:
                cam_pp = gcpp(input_tensor=tensor, targets=None)[0]

            # Metrics
            conc   = concentration(cam)
            faith  = faithfulness_drop(model, tensor, cam, pred_class)
            agr    = agreement(cam, cam_pp)
            t_row, t_col = top_activation(cam)
            quad   = quadrant(t_row, t_col)
            correct = int(pred_class == img_rec["true_label"])

            rows.append({
                "Filename":         img_rec["filename"],
                "True_Label":       img_rec["true_label"],
                "True_Name":        img_rec["true_name"],
                "Model":            MODEL_LABELS[model_name],
                "Pred_Prob_Pneumonia": round(pred_prob, 4),
                "Pred_Label":       pred_class,
                "Pred_Name":        "PNEUMONIA" if pred_class == 1 else "NORMAL",
                "Correct":          correct,
                "Concentration":    conc,
                "Faithfulness_Drop":faith,
                "GC_GCpp_Agreement":agr,
                "Top_Row":          t_row,
                "Top_Col":          t_col,
                "Focus_Quadrant":   quad,
            })

            #Progress
            if done % 50 == 0 or done == total:
                elapsed = time.time() - start
                eta     = (elapsed / done) * (total - done)
                print(f"  [{done:4d}/{total}] {img_rec['filename'][:30]:<30} "
                      f"| {MODEL_LABELS[model_name]:<14} "
                      f"| Conc={conc:.3f} Faith={faith:.3f} Agr={agr:.3f} "
                      f"| ETA: {eta:.0f}s")

        except Exception as e:
            print(f"  ❌ Error on {img_rec['filename']} / {model_name}: {e}")
            rows.append({
                "Filename":         img_rec["filename"],
                "True_Label":       img_rec["true_label"],
                "True_Name":        img_rec["true_name"],
                "Model":            MODEL_LABELS[model_name],
                "Pred_Prob_Pneumonia": None,
                "Pred_Label":       None,
                "Pred_Name":        "ERROR",
                "Correct":          None,
                "Concentration":    None,
                "Faithfulness_Drop":None,
                "GC_GCpp_Agreement":None,
                "Top_Row":          None,
                "Top_Col":          None,
                "Focus_Quadrant":   "ERROR",
            })

#Save CSV
os.makedirs("models", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s")
print(f"Saved: {OUTPUT_CSV}")
print(f"   Rows: {len(df)} ({len(image_records)} images × {len(loaded_models)} models)")
print(f"{'='*60}")

#Quick summary per model
print("\nQUICK SUMMARY (mean metrics per model, correct predictions only):")
print("-" * 60)
correct_df = df[df["Correct"] == 1].copy()
summary = correct_df.groupby("Model")[
    ["Concentration", "Faithfulness_Drop", "GC_GCpp_Agreement"]
].agg(["mean", "std"]).round(3)
print(summary.to_string())

print("\nQUICK SUMMARY (mean metrics per model, ALL predictions):")
print("-" * 60)
summary_all = df.groupby("Model")[
    ["Concentration", "Faithfulness_Drop", "GC_GCpp_Agreement"]
].agg(["mean", "std"]).round(3)
print(summary_all.to_string())

print("\nFocus quadrant distribution per model:")
print("-" * 60)
quad_dist = df.groupby(["Model", "Focus_Quadrant"]).size().unstack(fill_value=0)
print(quad_dist.to_string())
