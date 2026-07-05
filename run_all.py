"""
End-to-end driver: prepare data -> train 4 models -> evaluate -> saliency.

Usage:
    python run_all.py                 # full run (uses epoch counts in config.py)
    python run_all.py --smoke         # 2-epoch smoke test of the whole pipeline
    python run_all.py --skip-train    # only (re)evaluate + saliency on saved models

The saliency stage is launched as a *separate process* on purpose: it runs on
CPU (via CUDA_VISIBLE_DEVICES=-1 inside saliency.py) so its large Score-CAM /
Integrated-Gradients / SHAP batches never hit the small-GPU memory limit, and a
fresh process is the only way that env-var can take effect.
"""
import os
import sys
import argparse
import subprocess

import data_loader as D
import train
import evaluate
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 2-epoch run to validate the pipeline end-to-end")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse already-trained weights")
    args = ap.parse_args()

    cache = D.build_or_load_cache()
    class_weight = D.compute_class_weights(cache["y_train"])

    if not args.skip_train:
        head_e = 2 if args.smoke else C.HEAD_EPOCHS
        ft_e = 2 if args.smoke else C.FINE_TUNE_EPOCHS
        for name in C.MODEL_NAMES:
            train.train_one(name, cache, class_weight, head_e, ft_e)

    print("\n########## EVALUATION ##########")
    evaluate.evaluate_all()

    print("\n########## SALIENCY (separate CPU process) ##########")
    subprocess.run([sys.executable, os.path.join(C.PROJECT_DIR, "saliency.py")],
                   check=True)

    print("\nPipeline complete. See the 'outputs/' folder.")


if __name__ == "__main__":
    main()
