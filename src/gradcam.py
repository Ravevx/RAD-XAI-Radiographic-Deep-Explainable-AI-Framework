# src/gradcam.py
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_target_layer(model, model_name):
    """Return the last conv/attention layer for each backbone."""
    if model_name == "resnet18":
        return [model.layer4[-1]]
    elif model_name == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    elif model_name == "vit_tiny":
        return [model.blocks[-1].norm1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam_overlay(model, model_name, input_tensor, original_image_np):
    """
    Generate Grad-CAM heatmap overlay.

    Args:
        model: trained PyTorch model
        model_name: 'resnet18', 'densenet121', or 'vit_tiny'
        input_tensor: preprocessed tensor (1, 3, 224, 224)
        original_image_np: numpy (224, 224, 3) float32 in [0, 1]

    Returns:
        overlay: (224, 224, 3) uint8 numpy image
        grayscale_cam: (224, 224) float32 heatmap
    """
    target_layers = get_target_layer(model, model_name)

    reshape = vit_reshape_transform if model_name == "vit_tiny" else None

    with GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor.to(device),
            targets=None
        )[0]

    overlay = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    return overlay, grayscale_cam
