# evaluate_midas.py
import torch
import os
import json
import numpy as np
from metrics import compute_rmse, compute_abs_rel, compute_delta1


def make_eval_mask(
    gt: torch.Tensor,
    min_depth: float = 1e-6,
    border: int = 3,
    clip_percentiles=(1.0, 99.0),
):
    """
    Mask:
    - finite
    - gt > min_depth
    - usuń border
    - dodatkowo: odetnij skrajne percentyle głębi (żeby regresja nie była zdominowana outlierami)
    """
    gt = gt.float()
    H, W = gt.shape

    mask = torch.isfinite(gt) & (gt > float(min_depth))

    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    if clip_percentiles is not None:
        lo_p, hi_p = clip_percentiles
        vals = gt[mask]
        if vals.numel() > 1000:  # sensownie liczmy percentyle
            lo = torch.quantile(vals, lo_p / 100.0)
            hi = torch.quantile(vals, hi_p / 100.0)
            mask &= (gt >= lo) & (gt <= hi)

    return mask


def _ls_affine(x: torch.Tensor, y: torch.Tensor):
    """Rozwiąż y ≈ a*x + b."""
    X = torch.stack([x, torch.ones_like(x)], dim=1)  # (N,2)
    sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    return sol[0].item(), sol[1].item()


def align_affine_to_invdepth(pred: torch.Tensor, gt: torch.Tensor, mask=None, eps=1e-6):
    pred = pred.float()
    gt = gt.float()
    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)

    p = pred[mask].view(-1)
    g = gt[mask].view(-1)
    inv_g = 1.0 / torch.clamp(g, min=eps)

    # korelacja znaków (czasem MiDaS jest "odwrócony")
    if p.numel() > 10:
        c = torch.corrcoef(torch.stack([p, inv_g]))[0, 1]
        if torch.isfinite(c) and c < 0:
            p = -p

    a, b = _ls_affine(p, inv_g)
    return a, b


def apply_invdepth_alignment(pred: torch.Tensor, a: float, b: float, eps=1e-6):
    inv = a * pred.float() + b
    inv = torch.clamp(inv, min=eps)
    return 1.0 / inv


def align_affine_to_logdepth(pred: torch.Tensor, gt: torch.Tensor, mask=None, eps=1e-6):
    """
    log(gt) ≈ a*pred + b  ->  gt ≈ exp(a*pred + b)
    Lepsze dla szerokich zakresów (dalekie punkty).
    """
    pred = pred.float()
    gt = gt.float()
    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)

    p = pred[mask].view(-1)
    g = gt[mask].view(-1)
    y = torch.log(torch.clamp(g, min=eps))

    # jeśli korelacja ujemna, odwróć pred
    if p.numel() > 10:
        c = torch.corrcoef(torch.stack([p, y]))[0, 1]
        if torch.isfinite(c) and c < 0:
            p = -p

    a, b = _ls_affine(p, y)
    return a, b


def apply_logdepth_alignment(pred: torch.Tensor, a: float, b: float):
    return torch.exp(a * pred.float() + b)


def align_and_scale_best(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    """
    Spróbuj dwóch mapowań i wybierz to, które daje mniejszy AbsRel na masce.
    """
    a1, b1 = align_affine_to_invdepth(pred, gt, mask=mask)
    d1 = apply_invdepth_alignment(pred, a1, b1)
    absrel1 = compute_abs_rel(d1, gt, mask=mask)

    a2, b2 = align_affine_to_logdepth(pred, gt, mask=mask)
    d2 = apply_logdepth_alignment(pred, a2, b2)
    absrel2 = compute_abs_rel(d2, gt, mask=mask)

    if absrel2 < absrel1:
        return "logdepth", (a2, b2), d2
    return "invdepth", (a1, b1), d1


def make_object_range_mask(
    gt: torch.Tensor,
    border: int = 3,
    low_pct: float = 20.0,
    high_pct: float = 80.0,
    min_depth: float = 1e-6,
):
    """
    Maska ograniczająca metryki i skalowanie do zakresu,
    w którym znajduje się większość obiektów (bez tła).
    """
    gt = gt.float()
    mask = torch.isfinite(gt) & (gt > min_depth)

    # border
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    vals = gt[mask]
    if vals.numel() < 100:
        return mask  # fallback

    lo = torch.quantile(vals, low_pct / 100.0)
    hi = torch.quantile(vals, high_pct / 100.0)

    mask &= (gt >= lo) & (gt <= hi)
    return mask


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

            mask = make_object_range_mask(
                gt,
                low_pct=20.0,
                high_pct=80.0,
                border=3,
            )


            method, (a, b), pred_scaled = align_and_scale_best(pred, gt, mask)

            rmse = compute_rmse(pred_scaled, gt, mask=mask)
            absrel = compute_abs_rel(pred_scaled, gt, mask=mask)
            delta1 = compute_delta1(pred_scaled, gt, mask=mask)

            np.save(scaled_path, pred_scaled.cpu().numpy().astype("float32"))

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"method": method, "a": a, "b": b, "rmse": rmse, "absrel": absrel, "delta1": delta1},
                    f,
                    indent=4,
                )
def make_edge_object_mask(
    gt: torch.Tensor,
    border: int = 3,
    min_depth: float = 1e-6,
    grad_pct: float = 70.0,          # zostaw top 30% gradientów
    depth_pct=(10.0, 90.0),          # odetnij skrajne głębie
):
    """
    Maska "obiektowa" na bazie GT:
    - wybiera piksele z dużym gradientem głębi (krawędzie obiektów)
    - dodatkowo ogranicza depth do percentyli (żeby nie brać skrajnych ścian/rogów)
    """
    gt = gt.float()
    mask = torch.isfinite(gt) & (gt > min_depth)

    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    # percentyle głębi (opcjonalnie)
    vals = gt[mask]
    if vals.numel() > 1000 and depth_pct is not None:
        lo = torch.quantile(vals, depth_pct[0] / 100.0)
        hi = torch.quantile(vals, depth_pct[1] / 100.0)
        mask &= (gt >= lo) & (gt <= hi)

    # Sobel gradient
    gx = torch.zeros_like(gt)
    gy = torch.zeros_like(gt)

    gx[:, 1:-1] = (gt[:, 2:] - gt[:, :-2]) * 0.5
    gy[1:-1, :] = (gt[2:, :] - gt[:-2, :]) * 0.5

    grad = torch.sqrt(gx * gx + gy * gy)

    gvals = grad[mask]
    if gvals.numel() < 100:
        return mask

    thr = torch.quantile(gvals, grad_pct / 100.0)
    mask &= grad >= thr

    return mask


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    evaluate_all_in_output(root)
