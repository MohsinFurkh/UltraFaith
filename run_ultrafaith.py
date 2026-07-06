"""
UltraFaith end-to-end driver (cross-modality faithfulness benchmark).

Stages:
  1. (optional) train the 4 backbones on FETAL_PLANES_DB (6-class).
     BUS-BRA models are reused from the earlier run.
  2. run the faithfulness sweep for all 8 (modality x backbone) configs.
     Each config runs in its *own* subprocess so a single native crash
     (occasionally seen in the shap/tf CPU-GPU stack) cannot abort the whole
     benchmark - the driver simply moves on and benchmark.py aggregates
     whatever configs succeeded.
  3. aggregate -> Tables 1-3 and Figures 2-3 (benchmark.py).

Usage:
    python run_ultrafaith.py                 # full run
    python run_ultrafaith.py --skip-fetal    # reuse fetal weights
    python run_ultrafaith.py --skip-faith    # only (re)aggregate
"""
import os
import sys
import argparse
import subprocess

import config as C
import train


def _run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=C.PROJECT_DIR).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetal", action="store_true",
                    help="reuse already-trained FETAL weights")
    ap.add_argument("--skip-faith", action="store_true",
                    help="skip the faithfulness sweep (only aggregate)")
    args = ap.parse_args()

    # ---- 1. fetal training --------------------------------------------------
    if not args.skip_fetal:
        need = [b for b in C.MODEL_NAMES
                if not os.path.exists(C.weights_path(b, "_FETAL"))]
        if need:
            print("Training FETAL backbones:", need)
            train.train_fetal(need)
        else:
            print("All FETAL weights present; skipping training.")

    # ---- 2. faithfulness sweep (isolated subprocess per config) -------------
    if not args.skip_faith:
        for modality in C.MODALITY_NAMES:
            for backbone in C.MODEL_NAMES:
                wp = C.weights_path(backbone,
                                    C.MODALITIES[modality]["weight_suffix"])
                if not os.path.exists(wp):
                    print(f"[skip] no weights for {modality}/{backbone} ({wp})")
                    continue
                rc = _run([sys.executable, os.path.join(C.PROJECT_DIR,
                          "faithfulness.py"), modality, backbone])
                if rc != 0:
                    print(f"[warn] {modality}/{backbone} exited with code {rc}; "
                          f"continuing")

    # ---- 3. aggregate -------------------------------------------------------
    _run([sys.executable, os.path.join(C.PROJECT_DIR, "benchmark.py")])
    print("\nUltraFaith complete. See outputs/faithfulness/")


if __name__ == "__main__":
    main()
