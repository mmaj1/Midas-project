import os
import numpy as np
import cv2


def run_for_image(project_root: str, image_name: str):
    out_dir = os.path.join(project_root, "synthetic", "output")
    glebie_dir = os.path.join(project_root, "synthetic", "glebia")

    stem, _ = os.path.splitext(image_name)
    if "_" in stem:
        suffix = stem.split("_", 1)[1]
    else:
        suffix = stem

    rgb_path = os.path.join(out_dir, image_name)
    depth_pred_path = os.path.join(glebie_dir, f"{stem}_midas", "depth_midas_scaled.npy")
    depth_gt_path = os.path.join(out_dir, f"depth_gt_{suffix}.npy")

    if not os.path.isfile(rgb_path):
        print("Brak pliku obrazu.")
        return
    if not os.path.isfile(depth_pred_path):
        print("Brak przeskalowanej mapy MiDaS dla tego obrazu. Użyj opcji 1 w menu")
        return
    if not os.path.isfile(depth_gt_path):
        print("Brak ground truth dla tego obrazu.")
        return

    img = cv2.imread(rgb_path)
    depth_pred = np.load(depth_pred_path)
    depth_gt = np.load(depth_gt_path)

    window_name = f"Badanie odleglosci punktu: {image_name}. Esc/Q/0 - wyjscie"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < depth_pred.shape[0] and 0 <= x < depth_pred.shape[1]:
                d_pred = float(depth_pred[y, x])
                d_gt = float(depth_gt[y, x])
                print(f"Punkt ({x}, {y}) -> MiDaS = {d_pred:.3f}, GT = {d_gt:.3f}")

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

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q"), ord("0")):
            break

    cv2.destroyAllWindows()


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    name = input("Podaj nazwę pliku PNG: ").strip()
    if name and name != "0":
        run_for_image(project_root, name)


if __name__ == "__main__":
    main()
