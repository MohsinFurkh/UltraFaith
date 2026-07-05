"""
UltraFaith benchmark aggregation and figures (paper Sections 5-6).

Consumes the per-config outputs of faithfulness.py plus the trained models and
produces:

  Table 1  classification (Acc, macro-F1, AUROC) per modality x backbone
  Table 2  faithfulness (AUC_del, AUC_ins, F, DA) per method x modality
           (averaged over the four backbones)
  Table 3  cross-modality transfer: Kendall tau of per-method F rank
           (breast vs fetal), Pearson r between F and localisation rho,
           mean rho of the best-F method on BUS-BRA
  Fig 2    deletion / insertion curves per method (one backbone per modality)
  Fig 3    qualitative saliency with lesion-mask contour (BUS-BRA)
  extra    F heatmap (method x backbone), F-vs-rho scatter, DA bars

All tables are written as CSV + Markdown; all figures at 300 dpi.

Run:  python benchmark.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config as C
C.enable_gpu_memory_growth()
import data_loader as D
from models import build_model

plt.rcParams["savefig.dpi"] = C.PLOT_DPI
_MCOLORS = {"Grad-CAM": "#1f77b4", "Integrated Gradients": "#ff7f0e",
            "SHAP": "#2ca02c", "Score-CAM": "#d62728"}
FIG_BACKBONE = "EfficientNetB4"       # backbone shown in Figs 2 & 3


def _save(fig, fname):
    p = os.path.join(C.FAITH_DIR, fname)
    fig.savefig(p, dpi=C.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.basename(p))


# --------------------------------------------------------------------------- #
#  Table 1 : classification performance
# --------------------------------------------------------------------------- #
def classification_table():
    rows = []
    for modality in C.MODALITY_NAMES:
        spec = C.MODALITIES[modality]
        nc = spec["num_classes"]
        cache = (D.build_or_load_cache() if modality == "BUS-BRA"
                 else D.build_or_load_fetal_cache())
        Xte, yte = cache["X_test"], cache["y_test"].astype(int)
        for bb in C.MODEL_NAMES:
            wp = os.path.join(C.MODELS_DIR, f"{bb}{spec['weight_suffix']}.weights.h5")
            if not os.path.exists(wp):
                continue
            import tensorflow as tf
            tf.keras.backend.clear_session()
            model, _, _ = build_model(bb, num_classes=nc)
            model.load_weights(wp)
            out = model.predict(Xte, batch_size=C.batch_for(bb), verbose=0)
            if nc == 1:
                proba = out.ravel()
                pred = (proba >= 0.5).astype(int)
                auroc = roc_auc_score(yte, proba)
                f1 = f1_score(yte, pred)
            else:
                pred = out.argmax(1)
                try:
                    auroc = roc_auc_score(yte, out, multi_class="ovr",
                                          average="macro")
                except Exception:
                    auroc = np.nan
                f1 = f1_score(yte, pred, average="macro")
            rows.append({"Dataset": modality, "Backbone": bb,
                         "Accuracy": accuracy_score(yte, pred) * 100,
                         "Macro-F1": f1 * 100, "AUROC": auroc})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.FAITH_DIR, "table1_classification.csv"), index=False)
    _write_md(df.round(3), "table1_classification.md", "Table 1: Classification")
    print("\n=== Table 1: Classification performance ===")
    print(df.round(3).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
#  Load all per-config faithfulness CSVs
# --------------------------------------------------------------------------- #
def load_faith():
    files = glob.glob(os.path.join(C.FAITH_DIR, "faith_*.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------- #
#  Table 2 : faithfulness benchmark
# --------------------------------------------------------------------------- #
def faithfulness_table(faith):
    g = (faith.groupby(["modality", "method"])
         .agg(AUC_del=("AUC_del", "mean"),
              AUC_ins=("AUC_ins", "mean"),
              F=("F", "mean"),
              DA=("DA_hit", "mean"))
         .reset_index())
    g["DA"] = (g["DA"] * 100).round(0)
    # order methods as in the paper
    order = {m: i for i, m in enumerate(C.ATTRIBUTION_METHODS)}
    g = g.sort_values(["modality", "method"],
                      key=lambda s: s.map(order) if s.name == "method" else s)
    g.to_csv(os.path.join(C.FAITH_DIR, "table2_faithfulness.csv"), index=False)
    _write_md(g.round(3), "table2_faithfulness.md",
              "Table 2: Faithfulness (avg over backbones)")
    print("\n=== Table 2: Faithfulness benchmark (avg over backbones) ===")
    print(g.round(3).to_string(index=False))
    return g


# --------------------------------------------------------------------------- #
#  Table 3 : cross-modality transfer & faithfulness-localisation gap
# --------------------------------------------------------------------------- #
def transfer_table(faith, table2):
    rows = []
    mods = [m for m in C.MODALITY_NAMES if m in faith["modality"].unique()]
    if len(mods) == 2:
        piv = table2.pivot(index="method", columns="modality", values="F")
        piv = piv.reindex(C.ATTRIBUTION_METHODS)
        tau, p = kendalltau(piv[mods[0]].values, piv[mods[1]].values)
        rows.append(("Kendall tau of F rank (%s vs %s)" % (mods[0], mods[1]),
                     round(float(tau), 3)))

    bus = faith[(faith["modality"] == "BUS-BRA") & faith["rho"].notna()]
    if len(bus) > 3:
        r, p = pearsonr(bus["F"].values, bus["rho"].values)
        rows.append(("Pearson r: F vs localisation rho (BUS-BRA)",
                     round(float(r), 3)))
        # best-F method on breast and its mean rho
        best = (bus.groupby("method")["F"].mean().idxmax())
        mean_rho = bus[bus["method"] == best]["rho"].mean()
        rows.append((f"Mean localisation rho, best-F method ({best})",
                     round(float(mean_rho), 3)))

    df = pd.DataFrame(rows, columns=["Quantity", "Value"])
    df.to_csv(os.path.join(C.FAITH_DIR, "table3_transfer.csv"), index=False)
    _write_md(df, "table3_transfer.md", "Table 3: Cross-modality transfer")
    print("\n=== Table 3: Cross-modality transfer & F-rho gap ===")
    print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
#  Figure 2 : deletion / insertion curves
# --------------------------------------------------------------------------- #
def fig_curves():
    for modality in C.MODALITY_NAMES:
        npz = os.path.join(C.FAITH_DIR, f"curves_{modality}_{FIG_BACKBONE}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        ts = d["ts"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for m in C.ATTRIBUTION_METHODS:
            axes[0].plot(ts, d[f"{m}__del"], color=_MCOLORS[m], label=m)
            axes[1].plot(ts, d[f"{m}__ins"], color=_MCOLORS[m], label=m)
        axes[0].set_title("Deletion (lower AUC better)")
        axes[1].set_title("Insertion (higher AUC better)")
        for ax in axes:
            ax.set_xlabel("fraction of most-salient pixels")
            ax.set_ylabel("predicted-class confidence")
            ax.legend(fontsize=8)
        fig.suptitle(f"Deletion / Insertion - {modality} ({FIG_BACKBONE})",
                     y=1.02)
        _save(fig, f"fig2_curves_{modality}.png")


# --------------------------------------------------------------------------- #
#  Figure 3 : qualitative saliency with lesion-mask contour
# --------------------------------------------------------------------------- #
def _overlay(gray, heat, alpha=0.5):
    import matplotlib.cm as cm
    g = gray.squeeze() / 255.0
    rgb = np.stack([g, g, g], -1)
    hm = cm.get_cmap("jet")(np.clip(heat, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * hm, 0, 1)


def fig_qualitative():
    for modality in C.MODALITY_NAMES:
        npz = os.path.join(C.FAITH_DIR, f"qual_{modality}_{FIG_BACKBONE}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        imgs = d["img"]; masks = d["mask"]
        methods = C.ATTRIBUTION_METHODS
        cols = ["input"] + methods + (["lesion mask"]
                                      if C.MODALITIES[modality]["has_masks"] else [])
        n = len(imgs)
        fig, axes = plt.subplots(n, len(cols), figsize=(2.4 * len(cols), 2.5 * n))
        axes = np.atleast_2d(axes)
        for r in range(n):
            g = imgs[r].squeeze()
            axes[r, 0].imshow(g, cmap="gray")
            for ci, m in enumerate(methods, start=1):
                axes[r, ci].imshow(_overlay(imgs[r], d["map__" + m][r]))
                if C.MODALITIES[modality]["has_masks"] and masks[r].sum() > 0:
                    axes[r, ci].contour(masks[r], levels=[0.5],
                                        colors="lime", linewidths=1.0)
            if C.MODALITIES[modality]["has_masks"]:
                axes[r, -1].imshow(g, cmap="gray")
                if masks[r].sum() > 0:
                    axes[r, -1].contour(masks[r], levels=[0.5],
                                        colors="lime", linewidths=1.2)
            for ci in range(len(cols)):
                axes[r, ci].set_xticks([]); axes[r, ci].set_yticks([])
                if r == 0:
                    axes[r, ci].set_title(cols[ci], fontsize=9)
        fig.suptitle(f"Qualitative saliency - {modality} ({FIG_BACKBONE})",
                     y=1.01)
        fig.tight_layout()
        _save(fig, f"fig3_qualitative_{modality}.png")


# --------------------------------------------------------------------------- #
#  Extra summary figures
# --------------------------------------------------------------------------- #
def extra_figures(faith, table2):
    # F heatmap: method x backbone, one panel per modality
    mods = [m for m in C.MODALITY_NAMES if m in faith["modality"].unique()]
    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 4.5))
    axes = np.atleast_1d(axes)
    for ax, modality in zip(axes, mods):
        sub = faith[faith["modality"] == modality]
        piv = (sub.groupby(["method", "backbone"])["F"].mean()
               .unstack().reindex(C.ATTRIBUTION_METHODS))
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns,
                                                               rotation=30, ha="right")
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                        va="center", color="w", fontsize=8)
        ax.set_title(f"F by method x backbone - {modality}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "fig_F_heatmap.png")

    # F vs rho scatter (BUS-BRA)
    bus = faith[(faith["modality"] == "BUS-BRA") & faith["rho"].notna()]
    if len(bus) > 3:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for m in C.ATTRIBUTION_METHODS:
            s = bus[bus["method"] == m]
            ax.scatter(s["F"], s["rho"], s=14, alpha=0.5,
                       color=_MCOLORS[m], label=m)
        r, _ = pearsonr(bus["F"], bus["rho"])
        ax.set_xlabel("F (faithfulness consistency)")
        ax.set_ylabel(r"$\rho$ (saliency mass in lesion)")
        ax.set_title(f"Faithfulness vs. localisation (BUS-BRA)  r={r:.2f}")
        ax.legend(fontsize=8)
        _save(fig, "fig_F_vs_rho.png")

    # DA bars per method per modality
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = table2.pivot(index="method", columns="modality",
                       values="DA").reindex(C.ATTRIBUTION_METHODS)
    piv.plot.bar(ax=ax)
    ax.set_ylabel("Directional agreement (%)")
    ax.set_title("Directional agreement (k=20%) by method and modality")
    ax.set_ylim(0, 100); ax.legend(title="modality")
    plt.xticks(rotation=20, ha="right")
    _save(fig, "fig_DA_bars.png")


# --------------------------------------------------------------------------- #
def _write_md(df, fname, title):
    with open(os.path.join(C.FAITH_DIR, fname), "w") as f:
        f.write(f"### {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def main():
    classification_table()
    faith = load_faith()
    if faith is None:
        print("No faithfulness CSVs found - run faithfulness.py first.")
        return
    t2 = faithfulness_table(faith)
    transfer_table(faith, t2)
    fig_curves()
    fig_qualitative()
    extra_figures(faith, t2)
    print(f"\nUltraFaith benchmark artefacts saved under: {C.FAITH_DIR}")


if __name__ == "__main__":
    main()
