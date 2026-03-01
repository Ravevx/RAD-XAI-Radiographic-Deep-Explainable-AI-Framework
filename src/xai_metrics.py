import numpy as np
import torch


def iou_score(cam, threshold=0.5):
    binary_cam = (cam > threshold).astype(float)
    total_pixels = cam.shape[0] * cam.shape[1]
    activated = binary_cam.sum()
    concentration = 1.0 - (activated / total_pixels)
    return float(concentration)


def faithfulness_score(model, input_tensor, cam, device, target_class=None):
    model.eval()
    t = input_tensor.to(device)

    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0]
        if target_class is None:
            target_class = probs.argmax().item()
        original_prob = probs[target_class].item()

    threshold = np.percentile(cam, 50)
    mask = torch.tensor(
        (cam < threshold).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)
    mask = mask.expand_as(t)
    masked_input = t * mask

    with torch.no_grad():
        out_masked = model(masked_input)
        masked_prob = torch.softmax(out_masked, dim=1)[0][target_class].item()

    return original_prob, masked_prob, original_prob - masked_prob


def pointing_game_score(cam, top_n=1):
    flat_idx = np.argsort(cam.ravel())[::-1][:top_n]
    return [np.unravel_index(i, cam.shape) for i in flat_idx]


def gradcam_vs_gradcampp_agreement(cam1, cam2, threshold=0.5):
    b1 = (cam1 > threshold).astype(float)
    b2 = (cam2 > threshold).astype(float)
    intersection = (b1 * b2).sum()
    union = ((b1 + b2) > 0).sum()
    return float(intersection / union) if union > 0 else 0.0
