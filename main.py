import os
import sys
import json
import numpy as np
import torch
import subprocess

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "evaluation"))
sys.path.append(os.path.join(project_root, "midas_wrapper"))

from metrics import compute_rmse, compute_abs_rel, compute_delta1
#from evaluate_midas import align_scale
from run_midas import load_midas, run_midas_on_image, save_depth_png
from evaluate_midas import make_edge_object_mask, align_and_scale_best

def menu():
    print("\n1 - Przetwarzanie obrazów MiDaS")
    print("2 - Badanie punktu na obrazie")
    print("3 - Generowanie scen 3D")
    print("0 - Powrót do menu")
    return input("Wybierz opcję: ").strip()


def process_all_images():
    input_dir = os.path.join(project_root, "synthetic", "output")
    output_root = os.path.join(project_root, "synthetic", "glebia")
    os.makedirs(output_root, exist_ok=True)
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png") and f.startswith("rgb_")
    ]

    if not files:
        print("Brak plików PNG w synthetic/output")
        return

    midas, transform = load_midas()

    for fname in sorted(files):
        stem, _ = os.path.splitext(fname)
        if "_" in stem:
            suffix = stem.split("_", 1)[1]
        else:
            suffix = stem

        rgb_path = os.path.join(input_dir, fname)
        gt_path = os.path.join(input_dir, f"depth_gt_{suffix}.npy")
        if not os.path.isfile(gt_path):
            print(f"Brak pliku ground truth dla {fname}")
            continue

        work_dir = os.path.join(output_root, f"{stem}_midas")
        os.makedirs(work_dir, exist_ok=True)

        raw_path = os.path.join(work_dir, "depth_midas_raw.npy")
        depth_pred = run_midas_on_image(
            rgb_path,
            raw_path,
            midas=midas,
            transform=transform,
        )

        gt = torch.from_numpy(np.load(gt_path))
        gt = torch.nan_to_num(gt, nan=0.0)
        pred = torch.from_numpy(depth_pred)
        pred = torch.nan_to_num(pred, nan=0.0)

        mask = make_edge_object_mask(
            gt,
            border=3,
            grad_pct=70.0,
            depth_pct=(10.0, 90.0),
        )



        method, (a, b), pred_scaled = align_and_scale_best(pred, gt, mask)



        scaled_path = os.path.join(work_dir, "depth_midas_scaled.npy")
        pred_np = pred_scaled.cpu().numpy().astype("float32")
        np.save(scaled_path, pred_np)

        png_path = scaled_path.replace(".npy", ".png")
        save_depth_png(pred_np, png_path)


        rmse = compute_rmse(pred_scaled, gt, mask=mask)
        absrel = compute_abs_rel(pred_scaled, gt, mask=mask)
        delta1 = compute_delta1(pred_scaled, gt, mask=mask)

        metrics = {
            "method": method,
            "a": float(a),
            "b": float(b),
            "rmse": float(rmse),
            "absrel": float(absrel),
            "delta1": float(delta1),
        }

        metrics_path = os.path.join(work_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"Utworzono: {work_dir}")


def run_point_inspection():
    from select_point import run_for_image

    while True:
        name = input("Podaj nazwę pliku PNG (0 - powrót): ").strip()
        if name == "0":
            return
        if not name.lower().endswith(".png"):
            print("Podaj nazwę pliku z rozszerzeniem .png")
            continue

        full = os.path.join(project_root, "synthetic", "output", name)
        if not os.path.isfile(full):
            print("Plik nie istnieje.")
            continue

        run_for_image(project_root, name)
        return


def generate_scene():
    script_path = os.path.join(project_root, "synthetic", "render_scene.py")
    subprocess.run([sys.executable, script_path])


def convert_depth_to_png():
    script_path = os.path.join(project_root, "synthetic", "convert_depth_to_png.py")
    subprocess.run([sys.executable, script_path])


def main():
    while True:
        choice = menu()
        if choice == "1":
            process_all_images()
        elif choice == "2":
            run_point_inspection()
        elif choice == "3":
            generate_scene()
        elif choice == "0":
            continue
        else:
            print("Nieprawidłowy wybór.")


if __name__ == "__main__":
    main()