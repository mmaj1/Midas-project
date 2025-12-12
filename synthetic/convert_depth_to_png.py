import os
import numpy as np
import cv2

def normalize_depth(depth):
    depth = np.nan_to_num(depth, nan=0.0)
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)
    depth_norm = (depth - d_min) / (d_max - d_min)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    return depth_norm


def convert_folder(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            npy_path = os.path.join(folder, filename)
            depth = np.load(npy_path)
            depth_png = normalize_depth(depth)
            png_name = filename.replace(".npy", ".png")
            png_path = os.path.join(folder, png_name)
            cv2.imwrite(png_path, depth_png)
            print(png_path)


def convert_all_depth_maps():
    project_root = os.path.dirname(os.path.dirname(__file__))

    output_dir = os.path.join(project_root, "synthetic", "output")
    if os.path.isdir(output_dir):
        convert_folder(output_dir)

    glebia_root = os.path.join(project_root, "synthetic", "glebia")
    if os.path.isdir(glebia_root):
        for name in os.listdir(glebia_root):
            path = os.path.join(glebia_root, name)
            if os.path.isdir(path):
                convert_folder(path)


if __name__ == "__main__":
    convert_all_depth_maps()
