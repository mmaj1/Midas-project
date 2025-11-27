# main.py – możesz zostawić jak jest
"""
Główny skrypt:
 1. Generuje sztuczną scenę (synthetic/render_scene.py)
 2. Uruchamia MiDaS na wygenerowanym obrazie
 3. Liczy metryki (RMSE, AbsRel, δ<1.25)
"""

import os
import subprocess
import sys


def run_python(script_path):
    """Uruchamia inny skrypt Pythona jako proces."""
    print(f"\n=== Uruchamiam: {script_path} ===")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(script_path),
    )
    if result.returncode != 0:
        raise SystemExit(f"Błąd podczas uruchamiania {script_path}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    render_script = os.path.join(project_root, "synthetic", "render_scene.py")
    midas_script = os.path.join(project_root, "midas_wrapper", "run_midas.py")
    eval_script = os.path.join(project_root, "evaluation", "evaluate_midas.py")

    run_python(render_script)
    run_python(midas_script)
    run_python(eval_script)

    print("\nProjekt wykonany. Możesz teraz uruchomić:")
    print("  python evaluation/select_point.py")
    print("aby klikać punkt na obrazie i odczytywać odległość.")


if __name__ == "__main__":
    main()
