import torch
import os
import json
import numpy as np
from metrics import compute_rmse, compute_abs_rel, compute_delta1

def make_eval_mask(gt: torch.Tensor, max_depth=10.0, border=3):
    """
    Maskuje:
    - gt > 0
    - gt < max_depth
    - usuwa piksele przy krawędziach obrazu
    """
    H, W = gt.shape

    mask = torch.isfinite(gt)
    mask &= gt > 0
    mask &= gt < max_depth

    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    return mask

def align_scale(pred: torch.Tensor, gt: torch.Tensor, mask=None):
    pred = pred.float()
    gt = gt.float()

    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)

    p = pred[mask].view(-1)
    g = gt[mask].view(-1)

    p_mean = p.mean()
    g_mean = g.mean()

    p_centered = p - p_mean
    g_centered = g - g_mean

    a = (p_centered * g_centered).sum() / (p_centered ** 2).sum()
    b = g_mean - a * p_mean

    return a.item(), b.item()


def evaluate_all_in_output(project_root: str):
    out_dir = os.path.join(project_root, "synthetic", "output")

    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith("depth_midas_") and fname.endswith(".npy"):
            stem = fname[len("depth_midas_") : -4]
            gt_path = os.path.join(out_dir, f"depth_gt_{stem}.npy")
            pred_path = os.path.join(out_dir, fname)
            if not os.path.isfile(gt_path):
                continue

            scaled_path = os.path.join(out_dir, f"depth_midas_scaled_{stem}.npy")
            metrics_path = os.path.join(out_dir, f"metrics_{stem}.json")

            gt = torch.from_numpy(np.load(gt_path))
            pred = torch.from_numpy(np.load(pred_path))

            mask = make_eval_mask(gt, max_depth=10.0, border=3)

            a, b = align_scale(pred, gt, mask=mask)
            pred_scaled = a * pred + b


            rmse = compute_rmse(pred_scaled, gt, mask=mask)
            absrel = compute_abs_rel(pred_scaled, gt, mask=mask)
            delta1 = compute_delta1(pred_scaled, gt, mask=mask)


            np.save(scaled_path, pred_scaled.cpu().numpy().astype("float32"))

            metrics = {
                "a": a,
                "b": b,
                "rmse": rmse,
                "absrel": absrel,
                "delta1": delta1,
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    evaluate_all_in_output(root)
