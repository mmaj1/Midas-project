"""
  RMSE
  AbsRel
  δ < 1.25
"""

import torch


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def compute_rmse(pred, gt, mask=None):
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)
    pred = pred[mask]
    gt = gt[mask]

    mse = torch.mean((pred - gt) ** 2)
    return torch.sqrt(mse).item()


def compute_abs_rel(pred, gt, mask=None):
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)
    pred = pred[mask]
    gt = gt[mask]

    return torch.mean(torch.abs(pred - gt) / gt).item()


def compute_delta1(pred, gt, mask=None, threshold=1.25):
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if mask is None:
        mask = torch.isfinite(gt) & (gt > 0)
    pred = pred[mask]
    gt = gt[mask]

    ratio = torch.maximum(pred / gt, gt / pred)
    good = (ratio < threshold).float()
    return torch.mean(good).item()
