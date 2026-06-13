"""
Multi-Omics Cancer Classification — Full Evaluation Pipeline
6 cancer types: BRCA, LUAD, COAD, KIRC, LIHC, THCA

Experiments:
  1. Geneformer only
  2. MethylGPT only
  3. Early Fusion (concat embeddings)
  4. Late Fusion  (ensemble probabilities)
  5. Baseline     (PCA on raw features — shows value of foundation models)

Outputs saved to results/:
  metrics_summary.csv       — Acc / F1 / AUC per experiment
  per_class_metrics.csv     — Precision / Recall / F1 per cancer type
  umap_embeddings.png       — UMAP colored by cancer type
  roc_curves.png            — Per-cancer ROC curves for best model
  confusion_matrix.png      — Confusion matrix for best model
  shap_summary.png          — SHAP feature importance
  metrics_comparison.png    — Bar chart comparing all experiments
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, classification_report,
    roc_curve, auc,
)
import xgboost as xgb
import shap
import umap

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data/processed")
GENEFORMER_EMB = DATA_DIR / "geneformer_embeddings.npy"
METHYLGPT_EMB  = DATA_DIR / "methylgpt_embeddings.npy"
MANIFEST       = Path("data/manifests/matched_samples.csv")
METHYLATION_H5 = DATA_DIR / "tcga_methylation.h5ad"
RNA_H5         = DATA_DIR / "tcga_rna_seq.h5ad"
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
N_SPLITS     = 5
RANDOM_STATE = 42
CANCER_COLORS = {
    "TCGA-BRCA": "#e41a1c",
    "TCGA-LUAD": "#377eb8",
    "TCGA-COAD": "#4daf4a",
    "TCGA-KIRC": "#ff7f00",
    "TCGA-LIHC": "#984ea3",
    "TCGA-THCA": "#a65628",
}


# ── Model ──────────────────────────────────────────────────────────────────

def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ── Data loading ───────────────────────────────────────────────────────────

def load_data():
    manifest = pd.read_csv(MANIFEST)
    labels   = manifest["project"].values
    le       = LabelEncoder()
    y        = le.fit_transform(labels)

    gf_emb = np.load(GENEFORMER_EMB)
    mg_emb = np.load(METHYLGPT_EMB) if METHYLGPT_EMB.exists() else None

    print(f"Samples          : {gf_emb.shape[0]}")
    print(f"Geneformer dims  : {gf_emb.shape[1]}")
    if mg_emb is not None:
        print(f"MethylGPT dims   : {mg_emb.shape[1]}")
    else:
        print("MethylGPT        : not yet available")
    print(f"Classes          : {le.classes_}")
    return gf_emb, mg_emb, y, labels, le


def load_raw_features():
    """Load raw features for baseline comparison."""
    raw = {}
    if RNA_H5.exists():
        import anndata as ad
        adata = ad.read_h5ad(RNA_H5)
        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        # PCA to 128 dims (same as MethylGPT embedding size)
        pca = PCA(n_components=min(128, X.shape[1], X.shape[0]-1), random_state=RANDOM_STATE)
        raw["rna_pca"] = pca.fit_transform(StandardScaler().fit_transform(X))
        print(f"Raw RNA PCA      : {raw['rna_pca'].shape}")
    if METHYLATION_H5.exists():
        import anndata as ad
        adata = ad.read_h5ad(METHYLATION_H5)
        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        pca = PCA(n_components=min(128, X.shape[1], X.shape[0]-1), random_state=RANDOM_STATE)
        raw["meth_pca"] = pca.fit_transform(StandardScaler().fit_transform(X))
        print(f"Raw Meth PCA     : {raw['meth_pca'].shape}")
    return raw


# ── Cross-validation ───────────────────────────────────────────────────────

def run_kfold(X, y, le, label=""):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    acc_scores, f1_scores, auc_scores = [], [], []
    fold_models, fold_val_idx, fold_y_pred, fold_y_proba = [], [], [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_xgb()
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
        y_bin = label_binarize(y_val, classes=list(range(len(le.classes_))))
        auc_val = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

        acc_scores.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc_val)
        fold_models.append(model)
        fold_val_idx.append(val_idx)
        fold_y_pred.append(y_pred)
        fold_y_proba.append(y_proba)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.append(y_proba)

        print(f"  Fold {fold+1}: Acc={acc:.3f}  F1={f1:.3f}  AUC={auc_val:.3f}")

    print(f"  ── Mean ──  Acc={np.mean(acc_scores):.3f}±{np.std(acc_scores):.3f}  "
          f"F1={np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}  "
          f"AUC={np.mean(auc_scores):.3f}±{np.std(auc_scores):.3f}")

    return {
        "label":       label,
        "acc_mean":    np.mean(acc_scores),   "acc_std":    np.std(acc_scores),
        "f1_mean":     np.mean(f1_scores),    "f1_std":     np.std(f1_scores),
        "auc_mean":    np.mean(auc_scores),   "auc_std":    np.std(auc_scores),
        "models":      fold_models,
        "val_idx":     fold_val_idx,
        "y_pred":      fold_y_pred,
        "y_proba":     fold_y_proba,
        "all_y_true":  np.array(all_y_true),
        "all_y_pred":  np.array(all_y_pred),
        "all_y_proba": np.vstack(all_y_proba),
    }


def late_fusion(gf_result, mg_result, y, le):
    n_classes = len(le.classes_)
    acc_scores, f1_scores, auc_scores = [], [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    print(f"\n{'='*60}")
    print(f"  Experiment: Late Fusion (ensemble)")
    print(f"{'='*60}")

    for fold in range(N_SPLITS):
        val_idx   = gf_result["val_idx"][fold]
        y_val     = y[val_idx]
        avg_proba = (gf_result["y_proba"][fold] + mg_result["y_proba"][fold]) / 2.0
        y_pred    = np.argmax(avg_proba, axis=1)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
        y_bin   = label_binarize(y_val, classes=list(range(n_classes)))
        auc_val = roc_auc_score(y_bin, avg_proba, multi_class="ovr", average="macro")

        acc_scores.append(acc);  f1_scores.append(f1);  auc_scores.append(auc_val)
        all_y_true.extend(y_val);  all_y_pred.extend(y_pred);  all_y_proba.append(avg_proba)
        print(f"  Fold {fold+1}: Acc={acc:.3f}  F1={f1:.3f}  AUC={auc_val:.3f}")

    print(f"  ── Mean ──  Acc={np.mean(acc_scores):.3f}±{np.std(acc_scores):.3f}  "
          f"F1={np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}  "
          f"AUC={np.mean(auc_scores):.3f}±{np.std(auc_scores):.3f}")

    return {
        "label":       "Late Fusion",
        "acc_mean":    np.mean(acc_scores),  "acc_std":    np.std(acc_scores),
        "f1_mean":     np.mean(f1_scores),   "f1_std":     np.std(f1_scores),
        "auc_mean":    np.mean(auc_scores),  "auc_std":    np.std(auc_scores),
        "all_y_true":  np.array(all_y_true),
        "all_y_pred":  np.array(all_y_pred),
        "all_y_proba": np.vstack(all_y_proba),
    }


# ── Visualizations ─────────────────────────────────────────────────────────

def plot_umap(embeddings_dict, labels):
    n = len(embeddings_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    unique = sorted(set(labels))

    for ax, (name, X) in zip(axes, embeddings_dict.items()):
        print(f"  Computing UMAP for {name}...")
        reducer   = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(X)
        for cancer in unique:
            mask = labels == cancer
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=CANCER_COLORS.get(cancer, "#333333"),
                       label=cancer.replace("TCGA-", ""), s=22, alpha=0.8, edgecolors="none")
        ax.set_title(f"UMAP — {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=9, markerscale=1.5)
        ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    out = RESULTS_DIR / "umap_embeddings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(result, le, title="Best Model"):
    """Per-cancer ROC curves (One-vs-Rest) for the best experiment."""
    y_true  = result["all_y_true"]
    y_proba = result["all_y_proba"]
    n_classes = len(le.classes_)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cancer in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CANCER_COLORS.get(cancer, "#333"),
                lw=2, label=f"{cancer.replace('TCGA-','')} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    out = RESULTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrix(result, le, title="Best Model"):
    y_true = result["all_y_true"]
    y_pred = result["all_y_pred"]
    labels = [l.replace("TCGA-", "") for l in le.classes_]
    cm     = confusion_matrix(y_true, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def save_per_class_metrics(results, le):
    """Save per-cancer-type precision / recall / F1 for each experiment."""
    rows = []
    cancer_names = [c.replace("TCGA-", "") for c in le.classes_]
    for r in results:
        if "all_y_true" not in r:
            continue
        report = classification_report(
            r["all_y_true"], r["all_y_pred"],
            target_names=cancer_names, output_dict=True, zero_division=0
        )
        for cancer in cancer_names:
            rows.append({
                "Experiment": r["label"],
                "Cancer":     cancer,
                "Precision":  round(report[cancer]["precision"], 3),
                "Recall":     round(report[cancer]["recall"],    3),
                "F1":         round(report[cancer]["f1-score"],  3),
                "Support":    int(report[cancer]["support"]),
            })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "per_class_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-class metrics saved → {out}")
    print(df.pivot_table(index="Cancer", columns="Experiment", values="F1").round(3).to_string())
    return df


def plot_shap(X, y, label="Best Model"):
    print("  Computing SHAP values...")
    model = make_xgb()
    model.fit(X, y)
    sample_size = min(200, X.shape[0])
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:sample_size])

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values)

    importance = mean_abs_shap.mean(axis=0)
    top_idx    = np.argsort(importance)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([f"dim_{i}" for i in top_idx[::-1]], importance[top_idx[::-1]], color="#377eb8")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Top 20 Most Important Embedding Dimensions\n({label})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_metrics_comparison(results):
    labels  = [r["label"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("acc", "Accuracy"), ("f1", "F1 Score (macro)"), ("auc", "AUC-ROC (macro)")]
    colors  = ["#e41a1c", "#377eb8", "#4daf4a"]

    for ax, (metric, title), color in zip(axes, metrics, colors):
        means = [r[f"{metric}_mean"] for r in results]
        stds  = [r[f"{metric}_std"]  for r in results]
        bars  = ax.bar(labels, means, yerr=stds, color=color, alpha=0.8,
                       capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.1); ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Model Comparison — Multi-Omics Cancer Classification (6 types, 800 samples)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Multi-Omics Cancer Classification")
    print("  Foundation Models: Geneformer + MethylGPT")
    print("=" * 60)
    print("\nLoading data...")
    gf_emb, mg_emb, y, labels, le = load_data()

    results = []

    # ── Experiment 1: Geneformer only ──
    gf_result = run_kfold(gf_emb, y, le, label="Geneformer only")
    results.append(gf_result)

    # ── Experiments 2-4: require MethylGPT embeddings ──
    if mg_emb is not None:
        mg_result    = run_kfold(mg_emb, y, le, label="MethylGPT only")
        results.append(mg_result)

        early_emb    = np.concatenate([gf_emb, mg_emb], axis=1)
        early_result = run_kfold(early_emb, y, le, label="Early Fusion")
        results.append(early_result)

        lf_result = late_fusion(gf_result, mg_result, y, le)
        results.append(lf_result)
    else:
        print("\nNote: MethylGPT embeddings not found — running Geneformer-only for now")
        print("      Rerun after Bojana uploads methylgpt_embeddings.npy to Drive")

    # ── Experiment 5: Baseline (raw PCA) ──
    print("\nLoading raw features for baseline comparison...")
    raw = load_raw_features()
    if raw:
        if "rna_pca" in raw and "meth_pca" in raw:
            baseline_X  = np.concatenate([raw["rna_pca"], raw["meth_pca"]], axis=1)
            label_str   = "Baseline (PCA)"
        elif "rna_pca" in raw:
            baseline_X  = raw["rna_pca"]
            label_str   = "Baseline (RNA PCA)"
        else:
            baseline_X  = raw["meth_pca"]
            label_str   = "Baseline (Meth PCA)"
        baseline_result = run_kfold(baseline_X, y, le, label=label_str)
        results.append(baseline_result)
    else:
        print("  Skipping baseline (h5ad files not found locally)")

    # ── Save metrics summary ──
    print("\n── Saving results ──")
    rows = []
    for r in results:
        rows.append({
            "Model":    r["label"],
            "Accuracy": f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}",
            "F1 Score": f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}",
            "AUC-ROC":  f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print(f"\n{df_results.to_string(index=False)}")
    print(f"\nSaved → {RESULTS_DIR / 'metrics_summary.csv'}")

    # ── Per-class metrics ──
    save_per_class_metrics(results, le)

    # ── Pick best model for detailed plots ──
    best = max(results, key=lambda r: r["auc_mean"])
    print(f"\nBest model: {best['label']} (AUC={best['auc_mean']:.3f})")

    if best["label"] == "Geneformer only":
        best_X = gf_emb
    elif best["label"] == "MethylGPT only":
        best_X = mg_emb
    elif best["label"] == "Early Fusion" and mg_emb is not None:
        best_X = np.concatenate([gf_emb, mg_emb], axis=1)
    else:
        best_X = gf_emb

    # ── UMAP ──
    print("\n── Generating UMAP plots... ──")
    umap_embs = {"Geneformer": gf_emb}
    if mg_emb is not None:
        umap_embs["MethylGPT"]    = mg_emb
        umap_embs["Early Fusion"] = np.concatenate([gf_emb, mg_emb], axis=1)
    plot_umap(umap_embs, labels)

    # ── ROC curves ──
    print("\n── Generating ROC curves... ──")
    plot_roc_curves(best, le, title=best["label"])

    # ── Confusion matrix ──
    print("\n── Generating confusion matrix... ──")
    plot_confusion_matrix(best, le, title=best["label"])

    # ── SHAP ──
    print("\n── Generating SHAP plot... ──")
    plot_shap(best_X, y, label=best["label"])

    # ── Metrics comparison bar chart ──
    print("\n── Generating metrics comparison chart... ──")
    clean = [{k: v for k, v in r.items()
              if k not in ("models","val_idx","y_pred","y_proba","all_y_true","all_y_pred","all_y_proba")}
             for r in results]
    plot_metrics_comparison(clean)

    print(f"\n✓ All outputs saved to: {RESULTS_DIR}/")
    print("  Files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
