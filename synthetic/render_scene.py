"""
Prosty generator przykładowej sceny.

Tworzy:
 - obraz RGB 256x256 z jasnym kwadratem na ciemnym tle
 - mapę głębi:
      tło  = 5.0 (daleko)
      kwadrat = 2.0 (bliżej)

Zapis:
 - synthetic/output/rgb_0001.png
 - synthetic/output/depth_gt_0001.npy
"""

import os
import numpy as np
import cv2

# katalog synthetic/output obok tego pliku
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    print("=== render_scene.py ===")
    print("OUTPUT_DIR:", OUTPUT_DIR)

    # utworzenie katalogu, jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # rozmiar obrazu
    h, w = 256, 256

    # obraz RGB: ciemne tło
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = (40, 40, 40)  # ciemnoszary

    # jasny kwadrat w środku
    x1, y1 = 64, 64
    x2, y2 = 192, 192
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (220, 220, 220), thickness=-1)

    # mapa głębi: tło = 5.0, kwadrat = 2.0
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[y1:y2, x1:x2] = 2.0

    # ścieżki do zapisania
    rgb_path = os.path.join(OUTPUT_DIR, "rgb_0001.png")
    depth_path = os.path.join(OUTPUT_DIR, "depth_gt_0001.npy")

    # UWAGA: cv2 zapisuje w BGR, więc konwersja z RGB -> BGR
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(depth_path, depth)

    print(f"[render_scene] Zapisano RGB:   {rgb_path}")
    print(f"[render_scene] Zapisano depth: {depth_path}")


if __name__ == "__main__":
    main()
