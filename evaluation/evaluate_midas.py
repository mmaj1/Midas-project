# evaluation/evaluate_midas.py
"""
Porównanie depth MiDaS z depth ground truth.

 - wczytuje depth_gt_0001.npy (synthetic/output)
 - wczytuje depth_midas_0001.npy (synthetic/output)
 - dopasowuje skalę (a, b) w predykcji MiDaS
 - liczy RMSE, AbsRel, δ<1.25
 - zapisuje:
      - zeskalowaną głębię MiDaS do depth_midas_scaled_0001.npy
      - metryki do metrics_0001.json
"""

import os
import json
import numpy as np
import torch

from metrics import compute_rmse, compute_abs_rel, compute_delta1  # lokalny import :contentReference[oaicite:4]{index=4}


def align_scale(pred: torch.Tensor, gt: torch.Tensor, mask=None):
    """Dopasowanie liniowe: gt ≈ a * pred + b"""
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


def evaluate_single_pair(project_root: str):
    out_dir = os.path.join(project_root, "synthetic", "output")
    gt_path = os.path.join(out_dir, "depth_gt_0001.npy")
    pred_path = os.path.join(out_dir, "depth_midas_0001.npy")

    scaled_path = os.path.join(out_dir, "depth_midas_scaled_0001.npy")
    metrics_path = os.path.join(out_dir, "metrics_0001.json")

    gt = torch.from_numpy(np.load(gt_path))
    pred = torch.from_numpy(np.load(pred_path))

    # dopasowanie skali MiDaS -> skala ground truth
    a, b = align_scale(pred, gt)
    pred_scaled = a * pred + b

    # metryki
    rmse = compute_rmse(pred_scaled, gt)
    absrel = compute_abs_rel(pred_scaled, gt)
    delta1 = compute_delta1(pred_scaled, gt)

    print(f"[evaluate] Skala: a={a:.4f}, b={b:.4f}")
    print(f"[evaluate] RMSE   = {rmse:.4f}")
    print(f"[evaluate] AbsRel = {absrel:.4f}")
    print(f"[evaluate] δ<1.25 = {delta1:.4f}")

    # zapis zeskalowanej mapy głębi MiDaS
    np.save(scaled_path, pred_scaled.cpu().numpy().astype("float32"))
    print(f"[evaluate] Zapisano depth MiDaS (po skalowaniu) do: {scaled_path}")

    # zapis metryk do JSON
    metrics = {
        "a": a,
        "b": b,
        "rmse": rmse,
        "absrel": absrel,
        "delta1": delta1,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"[evaluate] Zapisano metryki do: {metrics_path}")

    return metrics


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    evaluate_single_pair(project_root)
