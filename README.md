# UltraFaith

**How Faithful Is the Heatmap? A Self-Referential Faithfulness Metric and Cross-Modality Benchmark of Post-Hoc Saliency for Ultrasound Classification**

Reference implementation and open benchmark. UltraFaith measures whether a
post-hoc saliency map is the *causal* basis of a classifier's decision — not
just whether it looks plausible — and does so **across two ultrasound
modalities** so that faithfulness rankings can be compared, not assumed.

It evaluates **four attribution methods** (Grad-CAM, Integrated Gradients,
GradientSHAP, Score-CAM) across **four backbones** (MobileNetV2, EfficientNetB4,
ResNet50, DenseNet121) on **two tasks**:

| Modality | Task | Classes | Pixel masks |
|----------|------|---------|-------------|
| **BUS-BRA** | breast lesion classification | 2 (benign / malignant) | ✓ (localisation ρ) |
| **FETAL_PLANES_DB** | fetal plane identification | 6 | ✗ |

All models use a **128×128×1 (grayscale)** input, expanded to 3 channels
in-graph so ImageNet backbones apply unchanged.

---

## The faithfulness metric

For each image the model's **own** predicted class ŷ is fixed, a Gaussian-blur
baseline *b* (σ=11, kernel 31) is built, pixels are ranked by the saliency map,
and progressively **deleted** (replaced by *b*) / **inserted** (revealed from
*b*) over 20 steps, tracking p<sub>ŷ</sub>:

- **∆faith** — signed confidence drop when the top-20% salient pixels are
  removed; positive = causally faithful direction.
- **DA** — directional-agreement rate: fraction of images with ∆faith > 0.
- **AUC<sub>del</sub>↓ / AUC<sub>ins</sub>↑** and **F = AUC<sub>ins</sub> −
  AUC<sub>del</sub>** — the primary ranking metric (high only when deletion
  *and* insertion agree).
- **ρ** — energy pointing-game: fraction of saliency mass inside the lesion mask
  (BUS-BRA only) — the clinical-localisation proxy.

## Headline results

| Method | Breast F | Fetal F | Breast DA | Fetal DA |
|--------|----------|---------|-----------|----------|
| Grad-CAM | **0.16** | **0.25** | 72% | 83% |
| Integrated Gradients | 0.11 | 0.23 | 75% | 79% |
| SHAP | 0.08 | 0.17 | 76% | 79% |
| Score-CAM | −0.03 | 0.20 | 57% | 82% |

- Faithfulness is **low and method-dependent**, and far lower on breast than
  fetal ultrasound; Score-CAM is even **anti-faithful** on breast (F < 0).
- The method **ranking** transfers across modalities (Kendall τ = +0.67) but the
  faithfulness **level** does not — a method acceptable on one modality can be
  near-useless on another.
- Computational faithfulness correlates **weakly** with clinical localisation
  (Pearson r(F, ρ) = 0.07): a map can pass every perturbation test while
  localising clinically irrelevant structure.

Aggregated tables and per-image scores are in [`results/`](results/).

---

## Installation

```bash
git clone https://github.com/MohsinFurkh/UltraFaith.git
cd UltraFaith
pip install -r requirements.txt
```
Tested with Python 3.7, TensorFlow 2.8 on a single CUDA GPU. `shap` pulls in
`torch`; on Windows the code imports it before TensorFlow to avoid a DLL
load-order clash.

## Data setup

Download the datasets and point the code at them via environment variables
(or place them one level above the repo as `../BUS-BRA` and
`../Fetal US Dataset`):

```bash
export BUSBRA_DIR=/path/to/BUS-BRA              # Images/, Masks/, bus_data.csv
export FETAL_DIR=/path/to/FETAL_PLANES_DB       # Images/, FETAL_PLANES_DB_data.csv
```
- **BUS-BRA**: <https://zenodo.org/records/8231412>
- **FETAL_PLANES_DB**: <https://zenodo.org/records/3904280>

## Quickstart

```bash
# 1) breast classification + explainability comparison (trains 4 backbones)
python run_all.py                 # add --smoke for a 2-epoch pipeline test

# 2) full cross-modality faithfulness benchmark
python run_ultrafaith.py          # trains fetal models, runs the 8-config sweep, aggregates
python run_ultrafaith.py --skip-fetal   # reuse fetal weights
python run_ultrafaith.py --skip-faith   # only re-aggregate tables/figures

# individual stages
python train.py                   # BUS-BRA;  python train.py --modality FETAL
python evaluate.py                # classification metrics + plots
python saliency.py                # Grad-CAM / Score-CAM / IG / SHAP panels
python faithfulness.py BUS-BRA EfficientNetB4    # one config
python benchmark.py               # Tables 1-3 + Figures 2-3
```

## Repository layout

| File | Purpose |
|------|---------|
| `config.py` | paths (env-overridable), hyper-parameters, faithfulness protocol, modality registry |
| `data_loader.py` | grayscale 128×128×1 loading, **patient-level** stratified split (no leakage), cached |
| `models.py` | the four backbones (grayscale→3ch, ImageNet transfer; sigmoid or softmax head) |
| `train.py` | two-phase transfer learning (warmup → fine-tune), class weights, per-modality |
| `evaluate.py` | classification metrics, confusion matrices, ROC/PR, computational + qualitative plots |
| `saliency.py` | Grad-CAM, Score-CAM, Integrated Gradients, SHAP visual panels |
| `faithfulness.py` | **the metric engine**: deletion/insertion, ∆faith, DA, F, localisation ρ |
| `benchmark.py` | Tables 1–3, cross-modality τ, F-vs-ρ, deletion/insertion curves, qualitative panels |
| `run_all.py` / `run_ultrafaith.py` | end-to-end drivers |
| `results/` | aggregated tables + per-image faithfulness scores (open benchmark) |

Every figure is saved at **300 dpi**. Trained weights and cached arrays are
regenerable and are not versioned (see `.gitignore`).

## Notes

- Faithfulness is compute-heavy; the sweep runs on a fixed random sample of
  `config.N_FAITH_IMAGES` (default 120) test images per configuration, and each
  config runs in an isolated subprocess so a single native crash cannot abort
  the benchmark.
- On small-VRAM GPUs the larger backbones use a reduced batch size
  (`config.BATCH_OVERRIDE`) and the saliency/faithfulness stages can fall back to
  CPU.

## Citation

```bibtex
@article{ultrafaith,
  title   = {How Faithful Is the Heatmap? A Self-Referential Faithfulness Metric
             and Cross-Modality Benchmark of Post-Hoc Saliency for Ultrasound
             Classification},
  author  = {Mukhtar, Sayima and Nazir, Azra},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
