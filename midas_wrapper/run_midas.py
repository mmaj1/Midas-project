# midas_wrapper/run_midas.py
"""
Uruchomienie MiDaS na obrazie synthetic/output/rgb_0001.png.

Zapisuje:
 - synthetic/output/depth_midas_0001.npy  (surowa predykcja MiDaS, bez skali)
"""

import os
import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_midas(model_type: str = "DPT_Large"):
    """
    Ładuje model MiDaS z torch.hub.
    Przy pierwszym uruchomieniu pobierze się z internetu.
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform


def run_midas_on_image(image_path: str, output_npy_path: str, model_type: str = "DPT_Large"):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    print(f"[run_midas] Ładuję model MiDaS ({model_type})...")
    midas, transform = load_midas(model_type)

    print(f"[run_midas] Wczytuję obraz: {image_path}")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # skalowanie do rozmiaru wejścia
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth_pred = prediction[0].cpu().numpy().astype("float32")
    np.save(output_npy_path, depth_pred)

    print(f"[run_midas] Zapisano predykcję głębi (MiDaS RAW): {output_npy_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    rgb_path = os.path.join(project_root, "synthetic", "output", "rgb_0001.png")
    out_path = os.path.join(project_root, "synthetic", "output", "depth_midas_0001.npy")

    run_midas_on_image(rgb_path, out_path)


if __name__ == "__main__":
    main()
