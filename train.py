# train.py  ← run this ONCE to train all 3 models
import os
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from src.utils import (
    build_model, TRANSFORM_TRAIN, TRANSFORM,
    score_test_set, compute_metrics, build_test_df
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODELS    = ["resnet18", "densenet121", "vit_tiny"]
EPOCHS    = 5
BATCH     = 32
LR        = 1e-4
TRAIN_DIR = "data/chest_xray/train"
TEST_DIR  = "data/chest_xray/test"

#Data
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=TRANSFORM_TRAIN)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=TRANSFORM)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=False)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
print(f"Classes: {train_ds.classes}")  # ['NORMAL', 'PNEUMONIA']

# Handle class imbalance
counts = [train_ds.targets.count(i) for i in range(2)]
weights = [1.0 / c for c in counts]
sample_weights = [weights[t] for t in train_ds.targets]
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                          num_workers=0, pin_memory=False)

all_metrics = {}

for model_name in MODELS:
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    model     = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Train Acc: {train_acc:.2%}")

    model.eval()
    test_df  = build_test_df(TEST_DIR)
    scored   = score_test_set(model, test_df)
    metrics  = compute_metrics(scored)
    all_metrics[model_name] = metrics

    print(f"\n  Results for {model_name}:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    torch.save(model.state_dict(), f"models/{model_name}.pth")
    scored.to_csv(f"models/{model_name}_test_scores.csv", index=False)
    print(f"  ✅ Saved models/{model_name}.pth")

pd.DataFrame(all_metrics).T.to_csv("models/all_metrics.csv")
print("\n✅ All done! models/all_metrics.csv saved.")
