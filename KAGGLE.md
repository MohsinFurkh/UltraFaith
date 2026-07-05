# Running UltraFaith end-to-end on Kaggle

Kaggle's T4/P100 (~16 GB) trains and evaluates the full benchmark far faster than
a small local GPU. This runs **everything** (train 8 models @224 → faithfulness
sweep → tables/figures) and produces one downloadable results bundle.

### Prerequisites
- Turn **GPU on** (Settings → Accelerator → GPU T4 x2 or P100).
- Add both datasets to the notebook (Add Input):
  - **BUS-BRA** — folder with `Images/`, `Masks/`, `bus_data.csv`
  - **FETAL_PLANES_DB** — folder with `Images/`, `FETAL_PLANES_DB_data.csv`
- Frozen splits (`busbra_split.json`, `fetal_split.json`) ship in the repo, so the
  train/val/test partition is identical to the local one — no test-set leakage.

---

### Cell 1 — clone + install legacy Keras 2 (Kaggle only offers TF 2.16+)
Kaggle's default TF 2.16+/Keras 3 (a) saves weights the local TF 2.8 can't read
and (b) breaks `shap`/`tf-keras-vis`. The `tf-keras` package + `TF_USE_LEGACY_KERAS=1`
force `tf.keras` to be **Keras 2**, fixing both. The verify line **must print
`tf.keras 2.x`** — if it prints `3.x`, stop and re-run this cell.
```python
%cd /kaggle/working
!rm -rf UltraFaith && git clone -q https://github.com/MohsinFurkh/UltraFaith.git
%cd UltraFaith
!pip install -q tf-keras "tf-keras-vis==0.8.5" shap tabulate
!TF_USE_LEGACY_KERAS=1 python -c "import tensorflow as tf; print('TF',tf.__version__,'| tf.keras',tf.keras.__version__)"
```

### Cell 2 — point at the datasets + shared env
```python
import os
# EDIT these two to match your Add-Input paths (use the folder holding Images/):
os.environ['BUSBRA_DIR'] = '/kaggle/input/bus-bra/BUS-BRA'
os.environ['FETAL_DIR']  = '/kaggle/input/fetal-planes-db/Fetal US Dataset'
os.environ['UF_IMG_SIZE'] = '224'
os.environ['UF_TAG']      = '_224'
os.environ['UF_L2']       = '1e-4'
os.environ['UF_STRONG_AUG'] = '1'
os.environ['UF_BIG_BATCH224'] = '16'             # large-VRAM batch
!ls "$BUSBRA_DIR" && echo '---' && ls "$FETAL_DIR"
```
**Every training/eval command below is prefixed with `TF_USE_LEGACY_KERAS=1`** so
the Keras-2 backend is guaranteed in each sub-process (and the ones it spawns).

### Cell 3 — train the 4 breast models @224
```python
!TF_USE_LEGACY_KERAS=1 UF_DROPOUT=0.4 UF_LABEL_SMOOTH=0.05 UF_UNFREEZE=60 python train.py --modality BUS-BRA
```

### Cell 4 — train the 4 fetal models @224
```python
!TF_USE_LEGACY_KERAS=1 UF_DROPOUT=0.3 UF_UNFREEZE=40 python train.py --modality FETAL
```

### Cell 5 — faithfulness sweep (8 configs) + benchmark tables/figures
```python
!TF_USE_LEGACY_KERAS=1 python run_ultrafaith.py --skip-fetal   # sweep + aggregate (children inherit the flag)
```

### Cell 6 — bundle results for download
```python
!cd outputs && zip -qr /kaggle/working/ultrafaith_224_results.zip faithfulness results models
print('Download /kaggle/working/ultrafaith_224_results.zip from the Output panel')
```

---

### What to download and where it goes
Download **`ultrafaith_224_results.zip`** from the Kaggle *Output* panel. It contains:
- `faithfulness/` — `table1/2/3_*.csv`, `fig2_curves_*`, `fig3_qualitative_*`,
  `fig_F_heatmap/…`, per-image `faith_*.csv`
- `results/` — per-model histories & compute JSON
- `models/` — the 8 `*_224*.weights.h5` checkpoints

Unzip it into the local project's `outputs/` (merging the folders). The tables +
figures are then plugged straight into the papers.

### Notes
- **The Cell 1 verify line must print `tf.keras 2.x`.** If it prints `3.x`, the
  legacy backend isn't active — the checkpoints would again save in a format the
  local TF 2.8 can't read and `shap`/`tf-keras-vis` would fail. Do not proceed
  until it shows 2.x (re-running Cell 1 usually fixes it).
- `TF_USE_LEGACY_KERAS=1` is prefixed on every command on purpose: an inline
  prefix guarantees the flag is in the sub-process environment *before* TensorFlow
  is imported, which `os.environ` set mid-notebook does not reliably achieve.
- `run_ultrafaith.py` launches the faithfulness configs as child processes; they
  inherit the flag from their parent, so the whole sweep stays on Keras 2.
- Runtime: roughly 1–2 h total on a single T4 (training dominates).
