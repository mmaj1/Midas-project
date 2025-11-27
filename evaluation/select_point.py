# evaluation/select_point.py
"""
Interaktywne wybieranie punktu na obrazie i odczyt odległości od kamery.

Używa:
 - synthetic/output/rgb_0001.png
 - synthetic/output/depth_midas_scaled_0001.npy  (MiDaS po skalowaniu)
 - synthetic/output/depth_gt_0001.npy            (ground truth)
"""

import os
import numpy as np
import cv2


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(project_root, "synthetic", "output")

    rgb_path = os.path.join(out_dir, "rgb_0001.png")
    depth_pred_path = os.path.join(out_dir, "depth_midas_scaled_0001.npy")
    depth_gt_path = os.path.join(out_dir, "depth_gt_0001.npy")

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.isfile(depth_pred_path):
        raise FileNotFoundError(
            f"Brak {depth_pred_path} – uruchom najpierw main.py, żeby policzyć MiDaS i skalowanie."
        )

    img = cv2.imread(rgb_path)
    depth_pred = np.load(depth_pred_path)  # H x W
    depth_gt = np.load(depth_gt_path)

    if depth_pred.shape != depth_gt.shape[:2]:
        print("Uwaga: rozmiar depth_pred i depth_gt się różni:", depth_pred.shape, depth_gt.shape)

    window_name = "Kliknij lewym przyciskiem, aby odczytać odległość (ESC lub Q, aby wyjść)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < depth_pred.shape[0] and 0 <= x < depth_pred.shape[1]:
                d_pred = float(depth_pred[y, x])
                d_gt = float(depth_gt[y, x])
                print(f"Punkt ({x}, {y}) -> MiDaS ≈ {d_pred:.3f},  GT = {d_gt:.3f}")

                # wizualizacja na obrazie
                disp = img.copy()
                cv2.circle(disp, (x, y), 4, (0, 0, 255), thickness=-1)
                text = f"{d_pred:.2f}"
                cv2.putText(
                    disp,
                    text,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, disp)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 512)
    cv2.setMouseCallback(window_name, on_mouse)

    # główna pętla GUI
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
