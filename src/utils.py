
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    accuracy_score, roc_curve
)
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def build_model(model_name):
    """Build a binary classification model."""
    if model_name == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif model_name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    elif model_name == "vit_tiny":
        m = timm.create_model("vit_tiny_patch16_224",pretrained=True,num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m.to(device)

def load_saved_model(model_name):
    """Load a model from models/ directory."""
    path = f"models/{model_name}.pth"
    if not os.path.exists(path):
        return None
    m = build_model(model_name)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

#Inference
def predict_single(model, image_path):
    """
    Predict on a single image file.
    Returns: prob (float) for PNEUMONIA class, input_tensor, orig_np
    """
    img = Image.open(image_path).convert("RGB")
    orig_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0][1].item()

    return prob, tensor, orig_np

def predict_pil(model, pil_image):
    """
    Predict on an already-loaded PIL image.
    Returns: prob (float), tensor, orig_np
    """
    orig_np = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0][1].item()

    return prob, tensor, orig_np

def build_test_df(test_dir="data/chest_xray/test"):
    """
    Walk test directory and return a dataframe with:
        path, true_label (0=NORMAL, 1=PNEUMONIA)
    """
    rows = []
    for label_name, label_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        folder = os.path.join(test_dir, label_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "path": os.path.join(folder, fname),
                    "filename": fname,
                    "true_label": label_idx,
                    "true_name": label_name
                })
    return pd.DataFrame(rows)

def score_test_set(model, test_df):
    """Run inference on all test images. Returns df with prob column."""
    probs = []
    for path in test_df["path"]:
        prob, _, _ = predict_single(model, path)
        probs.append(prob)
    df = test_df.copy()
    df["prob"] = probs
    return df

def compute_metrics(df, threshold=0.5):
    """Compute AUC, accuracy, sensitivity, specificity."""
    y_true = df["true_label"].values
    y_prob = df["prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn)
    }

def get_roc_curve(df):
    """Return (fpr, tpr, thresholds) for ROC curve."""
    y_true = df["true_label"].values
    y_prob = df["prob"].values
    return roc_curve(y_true, y_prob)
