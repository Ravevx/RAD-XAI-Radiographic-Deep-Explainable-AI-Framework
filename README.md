### RAD-XAI: Radiographic Deep Explainable-AI Framework
Explainable AI for Pneumonia Detection from Chest X-Rays

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This Project?

Most AI research asks one question: *"Is the model accurate?"*

This project asks a harder, more important question: **"Can we trust *why* the model made that decision?"**

In medicine, this difference is everything. An AI model can be 99% accurate and still be making decisions for the wrong reasons, and when it fails, you have no way to understand why or catch it in time.

We trained three deep learning models to detect pneumonia from chest X-rays, then used **XAI (Explainable AI)** tools to evaluate not just *what* each model decided but *how trustworthy its reasoning was*. Every prediction is accompanied by a Grad-CAM heatmap showing exactly where in the X-ray the model was looking, and three quality metrics that verify whether that heatmap can actually be trusted.

---

## What Problem Does It Solve?

Pneumonia is frequently missed in chest X-rays, especially in early stages and misdiagnosis can be life-threatening. AI can assist radiologists, but only if doctors can verify the AI's reasoning before acting on it. A black-box that says "Pneumonia: 98% confidence" is not clinically usable.

This project solves that by providing:

-  A pneumonia classifier achieving AUC above **0.98** across all three models
-  **Grad-CAM heatmaps** visual overlays showing where each model focused on the X-ray
-  **Three XAI quality metrics** verifying whether the heatmaps are genuine and consistent
-  **Multi-model comparison**: CNN baseline, medical CNN, and Vision Transformer
-  **Threshold analysis** across 99 operating points to find the safest clinical setting
-  **Case study analysis** across four clinical scenarios: correct catches, misses, and false alarms

---

## Models

| Model | Architecture | Role |
|-------|-------------|------|
| **ResNet-18** | Standard CNN | Baseline, widely used, well understood |
| **DenseNet-121** | Dense CNN | Validated in medical imaging (CheXNet architecture) |
| **ViT-Tiny** | Vision Transformer | Modern lightweight architecture tests XAI compatibility |

---

## XAI Metrics Explained

Three metrics judge whether a heatmap is trustworthy:

### Concentration ↑ (Higher = Better)
Is the model looking at one specific spot, or staring at everything at once?
Think of it as a spotlight, a focused spotlight on one actor is useful; a spotlight covering the whole stage tells you nothing.
- **1.0** = tight, precise spotlight 
- **0.5** = attention spread broadly
- **0.0** = looking everywhere equally, useless heatmap

### Faithfulness Drop ↑ (Higher = Better)
If you cover the highlighted region, does the model's confidence drop significantly?
- **1.0** = covering that region destroys model confidence - the heatmap is genuine 
- **0.0** = covering it changes nothing - the heatmap is misleading 
- **Negative** = covering it actually *increases* confidence - the model was being misled by that region

### GC/GC++ Agreement ↑ (Higher = Better)
Do Grad-CAM and Grad-CAM++ point at the same region when asked independently?
- **1.0** = both methods agree completely, stable, verifiable explanation 
- **0.0** = they point at completely different regions, explanation cannot be trusted 

---

## Key Findings

### Finding 1 - ResNet-18 is the most trustworthy model
Faithfulness **0.715** and Agreement **0.831**. When correct, faithfulness = 0.763. When wrong, faithfulness = 0.533 (+23% gap). Explanation quality tracks prediction quality - the most honest model for clinical audit.

### Finding 2 - DenseNet-121 is the most consistent explainer
Highest Agreement of all three: **0.885**. Faithfulness gap of **+0.439** between correct (0.701) and wrong (0.263) predictions. A faithfulness score below 0.30 reliably identifies wrong predictions - making it usable as a **live clinical safety flag**.

### Finding 3 - ViT-Tiny is accurate but unauditable
Concentration **0.900** (best of all models). But Agreement is **0.085** - near zero across the entire 624-image test set, in every scenario, whether the prediction was right or wrong. Gradient-based XAI methods (Grad-CAM, Grad-CAM++) are structurally incompatible with transformer architectures.

### Finding 4 - The FN Perception Failure (most important finding)
In the False Negative case (real pneumonia, all 3 models predicted NORMAL):
- All 3 models had faithfulness scores above **0.92** - the highest in the entire dataset
- All 3 independently focused on the **same lung region**
- All 3 were under **1% pneumonia confidence** - completely, confidently wrong

> A model can look at exactly the right place, faithfully use that region for its decision, and still fail to detect disease. This is a **perception failure, not an attention failure.** It is only detectable through XAI - accuracy metrics alone would never reveal it.

### Finding 5 - Default threshold (0.50) is unsafe for clinical use

| Threshold | Missed Pneumonia Cases | Recommendation |
|-----------|----------------------|----------------|
| **0.10** | 1–2 out of 390 | Recommended for clinical use |
| **0.50** (default) | 4–9 out of 390 | Not appropriate for medicine |
| **0.90** | 14–18 out of 390 | Misses too many cases |

---

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)**
> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |

---

## Project Structure

```
XAI_xray_analysis/
│
├── data/                          # Dataset
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── model/                         # Saved trained model weights
│   ├── resnet18_pneumonia.pth
│   ├── densenet121_pneumonia.pth
│   └── vit_tiny_pneumonia.pth
│   ├── xai_case_analysis.py           # Generates the 4 clinical case studies (TP/TN/FN/FP)
│   ├── xai_quality_analysis.py        # Computes XAI metrics for all 624 test images
│   ├── xai_quality_full.csv           # Output: XAI metrics - 1,872 rows (624 × 3 models)
│   ├── xai_case_study.csv             # Output: 4 case study images × 3 models
│ 
│ 
├── pages/                         # Multi-page Streamlit app pages
│   ├── 1_Model_Performance.py
│   ├── 2_XAI_Quality_Study.py
│   └── 3_Case_Studies.py
│
├── src/                           # Core source modules
│   ├── gradcam.py                 # Grad-CAM heatmap generation
│   ├── models.py                  # Model definitions and loaders
│   ├── utils.py                   # Shared utilities (image loading, preprocessing)
│   └── xai_metrics.py             # Concentration, Faithfulness, Agreement
│
├── app.py                         # Main Streamlit app entry point
├── evaluate_all.py                # Runs full evaluation across all models and images
├── train.py                       # Model training script
│ 
│
├── requirements.txt
└── README.md
```

---

## Installation

### Step 1 - Clone the repository

```bash
git clone https://github.com/yourusername/XAI_xray_analysis.git
cd XAI_xray_analysis
```

### Step 2 - Create a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Mac / Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
Pillow>=10.0.0
streamlit>=1.28.0
tqdm>=4.65.0
```

### Step 4 - Download the dataset

1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download and unzip the dataset
3. Place the folders inside `data/` so the structure matches above:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Usage

### 1. Train the models

```bash
python train.py
```

Trains ResNet-18, DenseNet-121, and ViT-Tiny on the training set. Saved weights go into the `model/` folder.

### 2. Evaluate model performance

```bash
python evaluate_all.py
```

Runs AUC, accuracy, sensitivity, and specificity across all 99 decision thresholds (0.01 to 0.99) for all three models.

### 3. Run XAI quality analysis

```bash
python xai_quality_analysis.py
```

Computes Concentration, Faithfulness Drop, and GC/GC++ Agreement for all 624 test images across all 3 models. Outputs `xai_quality_full.csv` (1,872 rows).

### 4. Run case study analysis

```bash
python xai_case_analysis.py
```

Selects representative images for all 4 clinical scenarios (TP, TN, FN, FP) and computes full XAI metrics. Outputs `xai_case_study.csv`.

### 5. Launch the interactive app

```bash
streamlit run app.py
```

Opens the multi-page Streamlit dashboard in your browser:

| Page | What it shows |
|------|--------------|
| **Model Performance** | AUC curves, accuracy, threshold analysis across all 3 models |
| **XAI Quality Study** | Concentration, Faithfulness, Agreement per model - correct vs wrong breakdown |
| **Case Studies** | Side-by-side heatmap comparison for TP, TN, FN, FP cases |

---

## Results Summary

| Model | AUC | Faithfulness | Agreement | Clinical Trust |
|-------|-----|-------------|-----------|----------------|
| ResNet-18 | **0.9851** | 0.715 | 0.831 | Trustworthy |
| DenseNet-121 | 0.9850 | 0.587 | **0.885** | Reliable |
| ViT-Tiny | 0.9823 | **0.737** | 0.085 | Unverifiable |

---

## What This Project Contributes

- Multi-architecture XAI comparison across 1,872 datapoints - not just selected examples
- Identification that ViT-Tiny is accurate but structurally unauditable with gradient-based XAI
- Discovery of the FN perception failure - high faithfulness heatmaps do not prevent missed diagnosis
- DenseNet-121 faithfulness as a clinical safety flag (threshold: 0.30)
- Clinically appropriate decision threshold recommendation (0.10) over the naive default (0.50)

---

## License

MIT License - free to use, modify, and distribute with attribution.

---

## Acknowledgements

- Dataset: Paul Mooney via Kaggle Chest X-Ray Images (Pneumonia)
- DenseNet-121 architecture inspired by CheXNet - Rajpurkar et al. (2017)
- Grad-CAM - Selvaraju et al., ICCV (2017)
- Grad-CAM++ - Chattopadhyay et al. (2018)
- Vision Transformer (ViT) - Dosovitskiy et al. (2020)

---

