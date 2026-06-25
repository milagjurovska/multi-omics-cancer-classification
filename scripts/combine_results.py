"""
Combines all experiment results into one final comparison table.
Run this AFTER:
  - train_classifier.py has finished         (results/metrics_summary.csv)
  - Mila's MOFA+ notebook has finished       (put baseline_metrics.csv in results/mofa/)
  - Bojana's autoencoder notebook finished   (autoencoder already included by train_classifier.py)

Usage:
    python3 scripts/combine_results.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
MOFA_CSV    = RESULTS_DIR / "mofa" / "baseline_metrics.csv"
MAIN_CSV    = RESULTS_DIR / "metrics_summary.csv"
OUTPUT_CSV  = RESULTS_DIR / "final_comparison.csv"
OUTPUT_PNG  = RESULTS_DIR / "final_comparison.png"


def load_main_results():
    if not MAIN_CSV.exists():
        print(f"ERROR: {MAIN_CSV} not found.")
        print("       Run 'python3 scripts/train_classifier.py' first.")
        return None
    df = pd.read_csv(MAIN_CSV)
    print(f"Loaded main results: {len(df)} experiments")
    return df


def load_mofa_results():
    if not MOFA_CSV.exists():
        print(f"WARNING: {MOFA_CSV} not found.")
        print("         Mila's MOFA+ notebook hasn't finished yet.")
        print("         Continuing without MOFA+ baseline...")
        return None

    df = pd.read_csv(MOFA_CSV)
    print(f"Loaded MOFA+ results: {len(df)} rows")

    rows = []
    for _, row in df.iterrows():
        # Handle Mila's actual column format: accuracy_mean / f1_macro_mean / auc_ovr_macro_mean
        if "accuracy_mean" in row:
            acc_mean = row["accuracy_mean"]
            acc_std  = row.get("accuracy_std", 0.0)
            f1_mean  = row.get("f1_macro_mean", row.get("f1_mean", 0.0))
            f1_std   = row.get("f1_macro_std",  row.get("f1_std",  0.0))
            auc_mean = row.get("auc_ovr_macro_mean", row.get("auc_mean", 0.0))
            auc_std  = row.get("auc_ovr_macro_std",  row.get("auc_std",  0.0))
            rows.append({
                "Model":    "MOFA+ (classical baseline)",
                "Accuracy": f"{acc_mean:.3f} ± {acc_std:.3f}",
                "F1 Score": f"{f1_mean:.3f} ± {f1_std:.3f}",
                "AUC-ROC":  f"{auc_mean:.3f} ± {auc_std:.3f}",
            })
            # Also extract raw single-modality baselines if present
            for col_prefix, label in [
                ("Methylation_only", "Raw Methylation Only"),
                ("RNA_only",         "Raw RNA Only"),
                ("RNA_plus_methylation", "Raw RNA + Methylation"),
            ]:
                if f"{col_prefix}_accuracy_mean" in row:
                    rows.append({
                        "Model":    label,
                        "Accuracy": f"{row[f'{col_prefix}_accuracy_mean']:.3f} ± 0.000",
                        "F1 Score": f"{row.get(f'{col_prefix}_f1_macro_mean', 0.0):.3f} ± 0.000",
                        "AUC-ROC":  f"{row.get(f'{col_prefix}_auc_ovr_macro_mean', 0.0):.3f} ± 0.000",
                    })
        else:
            # Fallback: generic column names
            rows.append({
                "Model":    row.get("label", row.get("Model", "MOFA+")),
                "Accuracy": row.get("Accuracy", "N/A"),
                "F1 Score": row.get("F1 Score", "N/A"),
                "AUC-ROC":  row.get("AUC-ROC",  "N/A"),
            })

    return pd.DataFrame(rows)


def plot_final_comparison(df):
    """Bar chart with all methods side by side."""
    models = df["Model"].tolist()

    def parse_mean(val):
        try:
            return float(str(val).split("±")[0].strip())
        except Exception:
            return 0.0

    acc  = [parse_mean(v) for v in df["Accuracy"]]
    f1   = [parse_mean(v) for v in df["F1 Score"]]
    aucs = [parse_mean(v) for v in df["AUC-ROC"]]

    # Color-code: baselines grey, single-modality blue/green, fusion red shades
    palette = []
    for m in models:
        if "Baseline" in m or "PCA" in m or "MOFA" in m:
            palette.append("#aaaaaa")
        elif "Geneformer" in m:
            palette.append("#377eb8")
        elif "MethylGPT" in m:
            palette.append("#4daf4a")
        elif "Early" in m:
            palette.append("#ff7f00")
        elif "Late" in m:
            palette.append("#e41a1c")
        elif "Auto" in m or "Deep" in m:
            palette.append("#984ea3")
        else:
            palette.append("#333333")

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.8), 7))
    bars1 = ax.bar(x - width, acc,  width, label="Accuracy",      color=palette, alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x,         f1,   width, label="F1 (macro)",    color=palette, alpha=0.6, edgecolor="white")
    bars3 = ax.bar(x + width, aucs, width, label="AUC-ROC",       color=palette, alpha=0.4, edgecolor="white")

    for bars, vals in [(bars1, acc), (bars2, f1), (bars3, aucs)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Final Model Comparison — Multi-Omics Cancer Classification\n"
                 "6 Cancer Types · 800 Samples · Foundation Models vs Classical Methods",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.axhline(y=0.9, color="red", linestyle="--", linewidth=0.8, alpha=0.4, label="0.9 threshold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUTPUT_PNG}")


def main():
    print("=" * 60)
    print("  Combining all results into final comparison table")
    print("=" * 60)

    main_df = load_main_results()
    if main_df is None:
        return

    mofa_df = load_mofa_results()

    # Combine: MOFA+ first (baseline), then main experiments
    if mofa_df is not None:
        final_df = pd.concat([mofa_df, main_df], ignore_index=True)
    else:
        final_df = main_df.copy()

    # Save
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(final_df.to_string(index=False))
    print(f"\nSaved → {OUTPUT_CSV}")

    # Plot
    print("\nGenerating final comparison chart...")
    plot_final_comparison(final_df)

    print("\nDone! Files in results/:")
    print(f"  final_comparison.csv  — full table")
    print(f"  final_comparison.png  — bar chart (put this in your report)")


if __name__ == "__main__":
    main()
