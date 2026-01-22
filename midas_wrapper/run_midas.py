import os
import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_depth_png(depth: np.ndarray, png_path: str):
    depth = np.nan_to_num(depth, nan=0.0)
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-8:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    import cv2
    cv2.imwrite(png_path, depth_norm)

def load_midas(model_type: str = "DPT_Large"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform


def run_midas_on_image(
    image_path: str,
    output_npy_path: str,
    model_type: str = "DPT_Large",
    midas=None,
    transform=None,
):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    if midas is None or transform is None:
        midas, transform = load_midas(model_type)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth_pred = prediction[0].cpu().numpy().astype("float32")
    np.save(output_npy_path, depth_pred)

    png_path = output_npy_path.replace(".npy", ".png")
    save_depth_png(depth_pred, png_path)

    return depth_pred



def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_dir = os.path.join(project_root, "synthetic", "output")
    midas, transform = load_midas()

    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(".png"):
            stem, _ = os.path.splitext(fname)
            image_path = os.path.join(input_dir, fname)
            out_path = os.path.join(input_dir, f"depth_midas_{stem}.npy")
            run_midas_on_image(image_path, out_path, midas=midas, transform=transform)


if __name__ == "__main__":
    main()
