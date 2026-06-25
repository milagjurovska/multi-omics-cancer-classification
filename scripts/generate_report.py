"""
Generates a publication-quality PDF research paper from all project results.
Run: python3 scripts/generate_report.py
Output: results/research_paper.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

RESULTS  = Path("results")
MOFA_DIR = RESULTS / "mofa"
OUTPUT   = RESULTS / "research_paper.pdf"

# ── Color palette ──────────────────────────────────────────────────────────
C = {
    "mofa":        "#aaaaaa",
    "raw_meth":    "#cccccc",
    "raw_rna":     "#bbbbbb",
    "raw_both":    "#999999",
    "geneformer":  "#377eb8",
    "methylgpt":   "#4daf4a",
    "early":       "#ff7f00",
    "late":        "#e41a1c",
    "autoencoder": "#984ea3",
    "accent":      "#2c3e50",
    "light":       "#ecf0f1",
    "title_bg":    "#1a252f",
}

CANCER_COLORS = {
    "TCGA-BRCA": "#e41a1c", "TCGA-LUAD": "#377eb8",
    "TCGA-COAD": "#4daf4a", "TCGA-KIRC": "#ff7f00",
    "TCGA-LIHC": "#984ea3", "TCGA-THCA": "#a65628",
}

def load_img(path):
    if Path(path).exists():
        return Image.open(path)
    return None

def show_img(ax, path, title=None):
    img = load_img(path)
    if img:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"[Figure not found:\n{Path(path).name}]",
                ha="center", va="center", fontsize=9, color="gray",
                transform=ax.transAxes)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

def add_page_number(fig, n):
    fig.text(0.5, 0.02, str(n), ha="center", fontsize=9, color="#888888")

def section_title(ax_or_fig, text, y=None, fontsize=14):
    if isinstance(ax_or_fig, plt.Figure):
        ax_or_fig.text(0.08, y, text, fontsize=fontsize, fontweight="bold",
                       color=C["accent"], transform=ax_or_fig.transFigure)
    else:
        ax_or_fig.set_title(text, fontsize=fontsize, fontweight="bold",
                            color=C["accent"], loc="left", pad=10)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title page
# ═══════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(C["title_bg"])

    # Top accent bar
    ax_top = fig.add_axes([0, 0.88, 1, 0.12])
    ax_top.set_facecolor("#2980b9")
    ax_top.axis("off")
    ax_top.text(0.5, 0.5,
                "MULTI-OMICS CANCER CLASSIFICATION",
                ha="center", va="center", fontsize=13,
                color="white", fontweight="bold",
                transform=ax_top.transAxes)

    ax = fig.add_axes([0, 0, 1, 0.88])
    ax.set_facecolor(C["title_bg"])
    ax.axis("off")

    ax.text(0.5, 0.80,
            "Foundation Model Embeddings for\nMulti-Omics Cancer Classification",
            ha="center", va="center", fontsize=22, color="white",
            fontweight="bold", transform=ax.transAxes, linespacing=1.4)

    ax.text(0.5, 0.66,
            "Integrating Geneformer and MethylGPT Representations\n"
            "with Fusion Strategies for TCGA Pan-Cancer Analysis",
            ha="center", va="center", fontsize=13, color="#bdc3c7",
            transform=ax.transAxes, linespacing=1.5)

    ax.plot([0.15, 0.85], [0.60, 0.60], color="#2980b9", linewidth=1.5,
            transform=ax.transAxes)

    ax.text(0.5, 0.54,
            "Mila Todorovska  ·  Bojana Andonova  ·  Sandra",
            ha="center", va="center", fontsize=12, color="white",
            transform=ax.transAxes)

    ax.text(0.5, 0.48,
            "Faculty of Computer Science and Engineering\nSs. Cyril and Methodius University",
            ha="center", va="center", fontsize=11, color="#bdc3c7",
            transform=ax.transAxes, linespacing=1.4)

    # Stats boxes
    stats = [("800", "Patients"), ("6", "Cancer Types"),
             ("5", "Experiments"), ("99.9%", "Best AUC")]
    for i, (val, lbl) in enumerate(stats):
        x = 0.12 + i * 0.20
        rect = FancyBboxPatch((x, 0.28), 0.15, 0.10,
                               boxstyle="round,pad=0.01",
                               facecolor="#2980b9", edgecolor="none",
                               transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.075, 0.345, val, ha="center", va="center",
                fontsize=16, color="white", fontweight="bold",
                transform=ax.transAxes)
        ax.text(x + 0.075, 0.295, lbl, ha="center", va="center",
                fontsize=9, color="#bdc3c7", transform=ax.transAxes)

    # Cancer type dots
    cancers = ["BRCA", "LUAD", "COAD", "KIRC", "LIHC", "THCA"]
    colors  = [C["accent"]] + list(CANCER_COLORS.values())[1:]
    full_c  = list(CANCER_COLORS.values())
    for i, (name, col) in enumerate(zip(cancers, full_c)):
        x = 0.10 + i * 0.135
        circ = plt.Circle((x + 0.045, 0.175), 0.022,
                           color=col, transform=ax.transAxes)
        ax.add_patch(circ)
        ax.text(x + 0.045, 0.135, name, ha="center", va="center",
                fontsize=8, color="#bdc3c7", transform=ax.transAxes)

    ax.text(0.5, 0.07, "TCGA · 2024 · Foundation Models · XGBoost · 5-Fold CV",
            ha="center", va="center", fontsize=9, color="#7f8c8d",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Abstract + Introduction
# ═══════════════════════════════════════════════════════════════════════════
def page_abstract(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Header stripe
    rect = FancyBboxPatch((0, 0.94), 1, 0.06, boxstyle="square,pad=0",
                           facecolor="#2980b9", edgecolor="none",
                           transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, 0.97, "Abstract & Introduction",
            ha="center", va="center", fontsize=14, color="white",
            fontweight="bold", transform=ax.transAxes)

    # Abstract box
    rect2 = FancyBboxPatch((0.05, 0.77), 0.90, 0.155,
                            boxstyle="round,pad=0.01",
                            facecolor="#eaf4fb", edgecolor="#2980b9", linewidth=1.5,
                            transform=ax.transAxes)
    ax.add_patch(rect2)
    ax.text(0.50, 0.935, "ABSTRACT", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2980b9",
            transform=ax.transAxes)

    abstract = (
        "Cancer type classification from multi-omics data is a critical challenge in precision oncology. "
        "This study evaluates the utility of two genomic foundation models — Geneformer (RNA-seq) and "
        "MethylGPT (DNA methylation) — for classifying six cancer types from The Cancer Genome Atlas (TCGA): "
        "BRCA, LUAD, COAD, KIRC, LIHC, and THCA (n=800). We compare single-modality embeddings against "
        "three fusion strategies (Early, Late, and Deep/Autoencoder Fusion) and a classical MOFA+ baseline. "
        "Geneformer achieves 97.7% accuracy (AUC=0.999) using only 1,152 dimensions vs. 60,616 raw genes. "
        "MethylGPT achieves 89.0% with 128 dimensions vs. 49,156 CpG probes. "
        "All fusion models exceed 96% accuracy with AUC=0.999. "
        "KEGG pathway analysis reveals enrichment in tumor suppressor and Wnt signaling pathways. "
        "Our results demonstrate that foundation model embeddings provide compact, biologically meaningful "
        "representations competitive with raw omics data."
    )
    ax.text(0.50, 0.845, abstract, ha="center", va="center",
            fontsize=8.5, color="#2c3e50", transform=ax.transAxes,
            wrap=True, multialignment="center",
            bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none"),
            linespacing=1.5)

    # Introduction
    def para(y, title, body, title_color="#2980b9"):
        ax.text(0.07, y, title, fontsize=10, fontweight="bold",
                color=title_color, transform=ax.transAxes)
        ax.text(0.07, y - 0.03, body, fontsize=8.5, color="#2c3e50",
                transform=ax.transAxes, linespacing=1.55,
                wrap=True, multialignment="left")

    ax.text(0.07, 0.745, "1. Introduction", fontsize=13, fontweight="bold",
            color=C["accent"], transform=ax.transAxes)

    para(0.705, "1.1  Motivation",
         "Cancer classification from multi-omics data enables personalized treatment decisions. Traditional\n"
         "machine learning methods require large feature sets (tens of thousands of genes or CpG probes),\n"
         "limiting interpretability and generalization. Foundation models trained on large biological datasets\n"
         "offer compact, transferable representations that may outperform or match raw feature approaches.")

    para(0.595, "1.2  Research Questions",
         "RQ1: Can foundation model embeddings classify cancer types with high accuracy?\n"
         "RQ2: Are multi-omics fusion strategies superior to single-modality approaches?\n"
         "RQ3: Do foundation model representations encode biologically meaningful signals?\n"
         "RQ4: How do foundation models compare to classical methods such as MOFA+?")

    para(0.485, "1.3  Contributions",
         "• First systematic comparison of Geneformer + MethylGPT on TCGA pan-cancer classification\n"
         "• Evaluation of three fusion strategies: Early (concatenation), Late (ensemble), Deep (autoencoder)\n"
         "• MOFA+ baseline providing a rigorous classical multi-omics comparison\n"
         "• KEGG pathway enrichment linking discriminative CpG probes to cancer biology\n"
         "• Full reproducible pipeline: data → embeddings → classification → biological interpretation")

    para(0.355, "1.4  Dataset",
         "We use TCGA level-3 data for 800 patients across 6 cancer types (≈133 samples/type).\n"
         "RNA-seq: 60,616 genes → Geneformer embeddings (1,152-dim).\n"
         "Methylation array: 49,156 CpG probes → MethylGPT embeddings (128-dim).\n"
         "All samples are matched by patient ID, ensuring paired RNA+methylation observations.")

    # Key numbers bar
    rect3 = FancyBboxPatch((0.05, 0.06), 0.90, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor="#1a252f", edgecolor="none",
                            transform=ax.transAxes)
    ax.add_patch(rect3)
    kv = [("800 patients", "6 cancer types", "49,156 CpG probes",
           "60,616 RNA genes", "5-fold cross-validation")]
    items = ["800 patients", "6 cancer types", "49,156 CpG probes",
             "60,616 RNA genes", "5-fold CV"]
    for i, item in enumerate(items):
        ax.text(0.10 + i * 0.185, 0.10, item, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                transform=ax.transAxes)

    add_page_number(fig, 2)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Methods architecture diagram
# ═══════════════════════════════════════════════════════════════════════════
def page_methods(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # Header
    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "2. Methods",
              ha="center", va="center", fontsize=14,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    # Architecture diagram
    ax = fig.add_axes([0.03, 0.45, 0.94, 0.46])
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_facecolor("#f8f9fa")

    def box(x, y, w, h, color, label, sublabel="", fontsize=9):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="white",
                               linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=7, color="#ecf0f1")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Input data
    box(0.1, 3.5, 1.8, 1.0, "#16a085", "RNA-seq", "800×60,616 genes")
    box(0.1, 2.2, 1.8, 1.0, "#8e44ad", "Methylation", "800×49,156 probes")

    # Foundation models
    box(2.3, 3.5, 2.0, 1.0, "#2980b9", "Geneformer", "Transformer · 1,152-dim")
    box(2.3, 2.2, 2.0, 1.0, "#27ae60", "MethylGPT", "Transformer · 128-dim")

    # Arrows input → models
    arrow(1.9, 4.0, 2.3, 4.0)
    arrow(1.9, 2.7, 2.3, 2.7)

    # Fusion strategies
    box(5.0, 4.2, 1.7, 0.8, "#e67e22", "Early Fusion", "concat → 1,280-dim")
    box(5.0, 3.2, 1.7, 0.8, "#c0392b", "Late Fusion", "ensemble proba")
    box(5.0, 2.2, 1.7, 0.8, "#8e44ad", "Deep Fusion", "autoencoder 64-dim")

    # Arrows embeddings → fusions
    for fy in [4.6, 3.6, 2.6]:
        arrow(4.3, 4.0, 5.0, fy)
        arrow(4.3, 2.7, 5.0, fy)

    # Classifier
    box(7.2, 3.1, 1.6, 1.0, "#2c3e50", "XGBoost", "5-fold StratifiedKV")

    # Arrows fusions → classifier
    for fy in [4.6, 3.6, 2.6]:
        arrow(6.7, fy, 7.2, 3.6)

    # Output
    box(9.1, 3.1, 0.8, 1.0, "#e74c3c", "Cancer\nType", fontsize=8)
    arrow(8.8, 3.6, 9.1, 3.6)

    # MOFA+ baseline (below)
    box(2.3, 0.5, 2.0, 0.9, "#7f8c8d", "MOFA+", "30 factors baseline")
    arrow(1.9, 4.0, 2.3, 0.95)
    arrow(1.9, 2.7, 2.3, 0.95)
    arrow(4.3, 0.95, 7.2, 3.3)

    ax.text(5.0, 0.2, "Figure 1. Full pipeline: raw omics → foundation model embeddings → "
            "fusion strategies → XGBoost classifier",
            ha="center", fontsize=8, color="#555", style="italic")

    # Methods text
    ax_t = fig.add_axes([0.05, 0.03, 0.90, 0.40])
    ax_t.axis("off")

    methods_text = {
        "2.1  Foundation Models": (
            "Geneformer [Theodoris et al., 2023] is a transformer pretrained on 29.9M single-cell transcriptomes. "
            "We extract the mean pooled hidden state (1,152-dim) from the final layer as the RNA embedding. "
            "MethylGPT [Fang et al., 2024] is a transformer trained on methylation arrays. We use the CLS token "
            "(128-dim) as the methylation embedding, selecting the top-2,048 most variable CpG probes as input."
        ),
        "2.2  Fusion Strategies": (
            "Early Fusion: Geneformer and MethylGPT embeddings are concatenated (1,280-dim) and fed to XGBoost. "
            "Late Fusion: Separate XGBoost classifiers are trained per modality; class probabilities are averaged. "
            "Deep Fusion: A cross-modal autoencoder learns a 64-dim shared latent space from both embeddings, "
            "trained with modality-specific reconstruction losses (AdamW, 200 epochs, CosineAnnealingLR)."
        ),
        "2.3  Classification & Evaluation": (
            "XGBoost (n_estimators=300, max_depth=6, lr=0.05) with 5-fold StratifiedKFold cross-validation. "
            "Metrics: Accuracy, macro F1, macro AUC-ROC (mean ± std across folds). "
            "MOFA+ (30 factors) trained strictly within each CV fold serves as the classical baseline. "
            "SHAP TreeExplainer provides feature importance for the best model."
        ),
    }

    y = 0.95
    for title, body in methods_text.items():
        ax_t.text(0, y, title, fontsize=10, fontweight="bold",
                  color="#2980b9", transform=ax_t.transAxes)
        y -= 0.06
        ax_t.text(0, y, body, fontsize=8.5, color="#2c3e50",
                  transform=ax_t.transAxes, linespacing=1.5,
                  wrap=True, multialignment="left")
        y -= 0.22

    add_page_number(fig, 3)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Results table + metrics comparison
# ═══════════════════════════════════════════════════════════════════════════
def page_results_table(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "3. Results — Performance Summary",
              ha="center", va="center", fontsize=14,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    # Load data
    df = pd.read_csv(RESULTS / "final_comparison.csv")

    def parse_mean(s):
        try: return float(str(s).split("±")[0].strip())
        except: return 0.0
    def parse_std(s):
        try: return float(str(s).split("±")[1].strip())
        except: return 0.0

    models = df["Model"].tolist()
    acc    = [parse_mean(v) for v in df["Accuracy"]]
    f1     = [parse_mean(v) for v in df["F1 Score"]]
    auc    = [parse_mean(v) for v in df["AUC-ROC"]]
    acc_s  = [parse_std(v)  for v in df["Accuracy"]]

    palette = []
    for m in models:
        if "MOFA" in m:          palette.append(C["mofa"])
        elif "Raw Meth" in m:    palette.append(C["raw_meth"])
        elif "Raw RNA Only" == m: palette.append(C["raw_rna"])
        elif "Raw RNA +" in m:   palette.append(C["raw_both"])
        elif "Geneformer" in m:  palette.append(C["geneformer"])
        elif "MethylGPT" in m:   palette.append(C["methylgpt"])
        elif "Early" in m:       palette.append(C["early"])
        elif "Late" in m:        palette.append(C["late"])
        elif "Auto" in m or "Deep" in m: palette.append(C["autoencoder"])
        else:                    palette.append("#333333")

    short = [m.replace("classical baseline", "baseline")
              .replace("(Autoencoder)", "")
              .replace("Raw ", "Raw\n")
              .replace("Deep Fusion ", "Deep\nFusion") for m in models]

    # Bar chart
    ax_bar = fig.add_axes([0.06, 0.52, 0.88, 0.38])
    x = np.arange(len(models))
    w = 0.27
    b1 = ax_bar.bar(x - w, acc, w, label="Accuracy", color=palette,
                    alpha=0.95, edgecolor="white", linewidth=0.8)
    b2 = ax_bar.bar(x,     f1,  w, label="F1 (macro)", color=palette,
                    alpha=0.65, edgecolor="white", linewidth=0.8)
    b3 = ax_bar.bar(x + w, auc, w, label="AUC-ROC", color=palette,
                    alpha=0.40, edgecolor="white", linewidth=0.8)

    for bar, val in zip(b1, acc):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Score", fontsize=10)
    ax_bar.set_title("Figure 2. Performance Comparison Across All Methods (5-fold CV)",
                     fontsize=10, fontweight="bold", pad=10)
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.axhline(0.9, color="red", lw=0.8, ls="--", alpha=0.4)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Separator
    ax_sep = fig.add_axes([0.05, 0.505, 0.90, 0.005])
    ax_sep.set_facecolor("#2980b9"); ax_sep.axis("off")

    # Table
    ax_tbl = fig.add_axes([0.03, 0.05, 0.94, 0.44])
    ax_tbl.axis("off")

    # Header
    cols = ["Model", "Accuracy", "F1 Score", "AUC-ROC", "Dims", "Category"]
    dims_map = {
        "MOFA+": "30 factors", "Raw Methylation": "49,156", "Raw RNA Only": "60,616",
        "Raw RNA +": "109,772", "Geneformer": "1,152", "MethylGPT": "128",
        "Early": "1,280", "Late": "1,152+128", "Deep": "64",
    }
    cat_map = {
        "MOFA+": "Classical", "Raw": "Raw Features",
        "Geneformer": "Foundation", "MethylGPT": "Foundation",
        "Early": "Fusion", "Late": "Fusion", "Deep": "Fusion",
    }

    col_widths = [0.30, 0.15, 0.15, 0.15, 0.13, 0.12]
    col_x = [0.01]
    for w_c in col_widths[:-1]:
        col_x.append(col_x[-1] + w_c)

    row_h  = 0.082
    header_y = 0.93

    # Header row
    hdr_rect = FancyBboxPatch((0, header_y - 0.01), 1.0, row_h + 0.01,
                               boxstyle="square,pad=0", facecolor="#2c3e50",
                               edgecolor="none", transform=ax_tbl.transAxes)
    ax_tbl.add_patch(hdr_rect)
    for j, (col, cx) in enumerate(zip(cols, col_x)):
        ax_tbl.text(cx + col_widths[j]/2, header_y + 0.025, col,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", transform=ax_tbl.transAxes)

    for i, row in df.iterrows():
        y_row = header_y - (i + 1) * row_h
        bg = "#eaf4fb" if i % 2 == 0 else "white"

        # Highlight best foundation model row
        m = str(row["Model"])
        if "Geneformer" in m:
            bg = "#d5e8d4"

        rect_r = FancyBboxPatch((0, y_row - 0.01), 1.0, row_h,
                                 boxstyle="square,pad=0", facecolor=bg,
                                 edgecolor="none", transform=ax_tbl.transAxes)
        ax_tbl.add_patch(rect_r)

        # Color dot
        dot_col = palette[i] if i < len(palette) else "#333"
        circ = plt.Circle((col_x[0] + 0.012, y_row + row_h/2 - 0.01),
                           0.013, color=dot_col, transform=ax_tbl.transAxes)
        ax_tbl.add_patch(circ)

        # Model name
        ax_tbl.text(col_x[0] + 0.03, y_row + row_h/2 - 0.01, m,
                    va="center", fontsize=8, color="#2c3e50",
                    transform=ax_tbl.transAxes)

        for j, col_name in enumerate(["Accuracy", "F1 Score", "AUC-ROC"]):
            val = str(row[col_name])
            ax_tbl.text(col_x[j+1] + col_widths[j+1]/2, y_row + row_h/2 - 0.01,
                        val, ha="center", va="center", fontsize=8,
                        color="#2c3e50", transform=ax_tbl.transAxes,
                        fontweight="bold" if "Geneformer" in m else "normal")

        # Dims
        dim_str = "–"
        for k, v in dims_map.items():
            if k in m:
                dim_str = v; break
        ax_tbl.text(col_x[4] + col_widths[4]/2, y_row + row_h/2 - 0.01,
                    dim_str, ha="center", va="center", fontsize=7.5,
                    color="#555", transform=ax_tbl.transAxes)
        # Category
        cat_str = "–"
        for k, v in cat_map.items():
            if k in m:
                cat_str = v; break
        ax_tbl.text(col_x[5] + col_widths[5]/2, y_row + row_h/2 - 0.01,
                    cat_str, ha="center", va="center", fontsize=7.5,
                    color="#555", transform=ax_tbl.transAxes)

    ax_tbl.text(0.5, -0.04,
                "Table 1. Full results. Green highlight = best foundation model. "
                "Dims = embedding dimensionality fed to classifier.",
                ha="center", fontsize=8, color="#555", style="italic",
                transform=ax_tbl.transAxes)

    add_page_number(fig, 4)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — UMAP visualizations
# ═══════════════════════════════════════════════════════════════════════════
def page_umap(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "3.1  UMAP Embedding Visualizations",
              ha="center", va="center", fontsize=14,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    ax_img = fig.add_axes([0.02, 0.35, 0.96, 0.56])
    show_img(ax_img, RESULTS / "umap_embeddings.png",
             "Figure 3. UMAP projections of all embedding spaces (800 patients, 6 cancer types)")

    ax_mofa = fig.add_axes([0.02, 0.05, 0.55, 0.28])
    show_img(ax_mofa, MOFA_DIR / "umap_mofa.png",
             "Figure 4. UMAP of MOFA+ latent factors (classical baseline)")

    # Interpretation panel
    ax_txt = fig.add_axes([0.60, 0.05, 0.38, 0.28])
    ax_txt.axis("off")
    ax_txt.text(0, 1.0, "Interpretation", fontsize=10, fontweight="bold",
                color="#2980b9", transform=ax_txt.transAxes, va="top")

    interp = [
        ("Geneformer", "#377eb8",
         "Strong cluster separation.\nRNA captures tissue identity well."),
        ("MethylGPT", "#4daf4a",
         "Clear separation for most types.\nSome BRCA/LUAD overlap."),
        ("Early Fusion", "#ff7f00",
         "Tightest clusters — best\ncombined representation."),
        ("Deep Fusion", "#984ea3",
         "Compact 64-dim space still\npreserves cancer structure."),
    ]
    for i, (name, col, desc) in enumerate(interp):
        y = 0.82 - i * 0.23
        circ = plt.Circle((0.03, y + 0.04), 0.025, color=col,
                           transform=ax_txt.transAxes)
        ax_txt.add_patch(circ)
        ax_txt.text(0.10, y + 0.04, name, fontsize=9, fontweight="bold",
                    color=col, va="center", transform=ax_txt.transAxes)
        ax_txt.text(0.10, y - 0.06, desc, fontsize=8, color="#444",
                    va="center", transform=ax_txt.transAxes, linespacing=1.4)

    add_page_number(fig, 5)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — ROC Curves + Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════
def page_roc_confusion(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "3.2  ROC Curves & Confusion Matrix (Best Model: Geneformer)",
              ha="center", va="center", fontsize=13,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    ax_roc = fig.add_axes([0.02, 0.50, 0.96, 0.41])
    show_img(ax_roc, RESULTS / "roc_curves.png",
             "Figure 5. Per-cancer ROC curves for best model (Geneformer, AUC=0.999)")

    ax_cm1 = fig.add_axes([0.02, 0.06, 0.55, 0.42])
    show_img(ax_cm1, RESULTS / "confusion_matrix.png",
             "Figure 6. Confusion matrix — Geneformer (foundation model)")

    ax_cm2 = fig.add_axes([0.58, 0.06, 0.40, 0.42])
    show_img(ax_cm2, MOFA_DIR / "baseline_confusion_matrix.png",
             "Figure 7. Confusion matrix — MOFA+ (classical baseline)")

    add_page_number(fig, 6)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — SHAP + Per-class metrics
# ═══════════════════════════════════════════════════════════════════════════
def page_shap(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "3.3  Feature Importance (SHAP) & Per-Class Performance",
              ha="center", va="center", fontsize=13,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    ax_shap = fig.add_axes([0.02, 0.50, 0.96, 0.41])
    show_img(ax_shap, RESULTS / "shap_summary.png",
             "Figure 8. Top 20 most important Geneformer embedding dimensions (mean |SHAP|)")

    # Per-class metrics table
    ax_tbl = fig.add_axes([0.03, 0.05, 0.94, 0.42])
    ax_tbl.axis("off")
    ax_tbl.text(0, 0.97, "Figure 9. Per-Cancer F1 Scores Across All Methods",
                fontsize=10, fontweight="bold", color="#2c3e50",
                transform=ax_tbl.transAxes)

    per_raw  = pd.read_csv(RESULTS / "per_class_metrics.csv")
    per = per_raw.pivot_table(index="Cancer", columns="Experiment", values="F1")
    cancers  = per.index.tolist()
    exp_cols = per.columns.tolist()

    col_w = 0.85 / len(exp_cols)
    row_h = 0.11

    # Header
    hdr = FancyBboxPatch((0, 0.82), 1.0, 0.12, boxstyle="square,pad=0",
                          facecolor="#2c3e50", edgecolor="none",
                          transform=ax_tbl.transAxes)
    ax_tbl.add_patch(hdr)
    ax_tbl.text(0.07, 0.88, "Cancer", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                transform=ax_tbl.transAxes)
    for j, col in enumerate(exp_cols):
        cx = 0.15 + j * col_w
        ax_tbl.text(cx + col_w/2, 0.88,
                    col.replace(" ", "\n").replace("(", "\n("),
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white",
                    transform=ax_tbl.transAxes)

    cancer_colors_list = list(CANCER_COLORS.values())
    for i, cancer in enumerate(cancers):
        y_row = 0.82 - (i + 1) * row_h
        bg = "#f8f9fa" if i % 2 == 0 else "white"
        r = FancyBboxPatch((0, y_row), 1.0, row_h, boxstyle="square,pad=0",
                           facecolor=bg, edgecolor="none",
                           transform=ax_tbl.transAxes)
        ax_tbl.add_patch(r)
        col_c = cancer_colors_list[i] if i < len(cancer_colors_list) else "#333"
        circ = plt.Circle((0.025, y_row + row_h/2), 0.018, color=col_c,
                           transform=ax_tbl.transAxes)
        ax_tbl.add_patch(circ)
        ax_tbl.text(0.07, y_row + row_h/2, cancer.replace("TCGA-", ""),
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color=col_c, transform=ax_tbl.transAxes)
        for j, col in enumerate(exp_cols):
            val = float(per.at[cancer, col])
            cx = 0.15 + j * col_w
            color = "#1a7a2e" if val >= 0.97 else "#2c3e50" if val >= 0.90 else "#c0392b"
            ax_tbl.text(cx + col_w/2, y_row + row_h/2,
                        f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold",
                        transform=ax_tbl.transAxes)

    add_page_number(fig, 7)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 — KEGG pathway analysis
# ═══════════════════════════════════════════════════════════════════════════
def page_kegg(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "3.4  KEGG Pathway Enrichment Analysis",
              ha="center", va="center", fontsize=14,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    # Try to load KEGG plot from results
    kegg_path = RESULTS / "kegg_top_pathways.png"
    if not kegg_path.exists():
        kegg_path = RESULTS / "mofa" / "kegg_top_pathways.png"

    ax_kegg = fig.add_axes([0.02, 0.42, 0.96, 0.49])
    show_img(ax_kegg, kegg_path,
             "Figure 10. Top 20 KEGG pathways enriched among discriminative CpG probes")

    # Top probes table
    ax_t = fig.add_axes([0.03, 0.05, 0.94, 0.35])
    ax_t.axis("off")
    ax_t.text(0, 0.97, "Figure 11. Top Discriminative CpG Probes (by ANOVA F-statistic)",
              fontsize=10, fontweight="bold", color="#2c3e50",
              transform=ax_t.transAxes)

    probes_path = RESULTS / "top_discriminative_probes.csv"
    if probes_path.exists():
        probes_df = pd.read_csv(probes_path).head(12)
        cols = probes_df.columns.tolist()[:4]
        col_w = 0.90 / len(cols)

        hdr = FancyBboxPatch((0, 0.80), 1.0, 0.14, boxstyle="square,pad=0",
                              facecolor="#2c3e50", edgecolor="none",
                              transform=ax_t.transAxes)
        ax_t.add_patch(hdr)
        for j, col in enumerate(cols):
            ax_t.text(j * col_w + col_w/2, 0.87, col,
                      ha="center", va="center", fontsize=8,
                      fontweight="bold", color="white", transform=ax_t.transAxes)

        for i, (_, row) in enumerate(probes_df.iterrows()):
            y_r = 0.80 - (i + 1) * 0.065
            bg = "#eaf4fb" if i % 2 == 0 else "white"
            r = FancyBboxPatch((0, y_r), 1.0, 0.065, boxstyle="square,pad=0",
                               facecolor=bg, edgecolor="none",
                               transform=ax_t.transAxes)
            ax_t.add_patch(r)
            for j, col in enumerate(cols):
                val = row[col]
                if isinstance(val, float):
                    val = f"{val:.4f}"
                ax_t.text(j * col_w + col_w/2, y_r + 0.032, str(val),
                          ha="center", va="center", fontsize=8,
                          color="#2c3e50", transform=ax_t.transAxes)
    else:
        ax_t.text(0.5, 0.5, "KEGG results not found", ha="center",
                  color="gray", transform=ax_t.transAxes)

    ax_t.text(0.5, -0.06,
              "ANOVA F-statistic identifies probes most discriminative across 6 cancer types. "
              "These probes are mapped to gene names via Illumina 450K annotation for KEGG enrichment.",
              ha="center", fontsize=8, color="#555", style="italic",
              transform=ax_t.transAxes)

    add_page_number(fig, 8)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 9 — Discussion + Conclusion
# ═══════════════════════════════════════════════════════════════════════════
def page_discussion(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_facecolor("#2980b9"); ax_h.axis("off")
    ax_h.text(0.5, 0.5, "4. Discussion & 5. Conclusion",
              ha="center", va="center", fontsize=14,
              color="white", fontweight="bold", transform=ax_h.transAxes)

    ax = fig.add_axes([0.06, 0.04, 0.88, 0.88])
    ax.axis("off")

    def section(y, title, body, color="#2980b9"):
        ax.text(0, y, title, fontsize=11, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0, y - 0.045, body, fontsize=8.8, color="#2c3e50",
                transform=ax.transAxes, linespacing=1.6,
                multialignment="left")

    section(0.96, "4.1  Foundation Models vs. Raw Features",
            "Geneformer (1,152-dim) matches raw RNA features (60,616-dim) within 1.8% accuracy "
            "(97.7% vs. 99.5%), representing a 52× dimensionality reduction. This demonstrates that "
            "pretraining on large single-cell corpora encodes tissue-discriminative signals without "
            "task-specific supervision. MethylGPT (128-dim) is 10% below raw methylation (89.0% vs. 99.0%), "
            "reflecting stronger information compression — methylation-specific context is harder to distill "
            "into 128 dimensions.")

    section(0.76, "4.2  Fusion Strategy Analysis",
            "Early Fusion (97.5%) and Late Fusion (97.2%) perform comparably to Geneformer alone (97.7%), "
            "suggesting that MethylGPT's weaker signal does not substantially improve the already strong RNA "
            "representation. Deep Fusion via autoencoder (96.6%) compresses both modalities into 64 dimensions "
            "with a small accuracy cost, providing the most compact representation. The near-identical AUC=0.999 "
            "across all fusion methods indicates that the discriminative ceiling is reached by Geneformer alone "
            "for this 6-class task — fusion benefits would likely be more pronounced for harder subtypes.")

    section(0.55, "4.3  Classical Baseline Comparison",
            "MOFA+ achieves 97.6% accuracy — virtually identical to Geneformer (97.7%). However, MOFA+ "
            "requires fitting a matrix factorization model on each CV fold and is sensitive to the number of "
            "factors. Foundation model embeddings are pre-computed once and reused, making them substantially "
            "more scalable. Raw single-modality features (RNA: 99.5%, Methylation: 99.0%) outperform all "
            "foundation models, consistent with the relative simplicity of tissue-of-origin classification "
            "compared to more challenging intra-cancer subtype tasks.")

    section(0.34, "4.4  Biological Interpretation",
            "KEGG enrichment of the top 500 discriminative CpG probes reveals pathways including Wnt signaling, "
            "PI3K-Akt, DNA repair, and cell cycle regulation — all established cancer driver mechanisms. "
            "SHAP analysis of Geneformer embeddings shows that a small subset of embedding dimensions drives "
            "classification, suggesting that the foundation model concentrates cancer-relevant information "
            "in interpretable dimensions.")

    # Conclusion box
    rect = FancyBboxPatch((0, 0.02), 1.0, 0.26,
                           boxstyle="round,pad=0.01",
                           facecolor="#1a252f", edgecolor="none",
                           transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, 0.265, "5. Conclusion", ha="center", fontsize=12,
            fontweight="bold", color="#2980b9", transform=ax.transAxes)
    ax.text(0.5, 0.22,
            "We present a comprehensive evaluation of Geneformer and MethylGPT foundation model\n"
            "embeddings for TCGA pan-cancer classification across 6 cancer types (n=800).\n\n"
            "Key findings:\n"
            "  ✦  Geneformer achieves 97.7% accuracy with 52× fewer dimensions than raw RNA features\n"
            "  ✦  All methods reach AUC=0.999, confirming strong separability of these cancer types\n"
            "  ✦  Fusion strategies do not substantially improve over Geneformer alone for this task\n"
            "  ✦  MOFA+ (97.6%) matches foundation models, validating both approaches\n"
            "  ✦  KEGG enrichment links discriminative probes to known cancer-driver pathways\n\n"
            "Foundation models provide biologically meaningful, compact representations that enable\n"
            "efficient multi-omics integration without task-specific training.",
            ha="center", va="top", fontsize=9, color="white",
            transform=ax.transAxes, linespacing=1.6)

    add_page_number(fig, 9)
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Generating research paper PDF...")
    with PdfPages(OUTPUT) as pdf:
        meta = pdf.infodict()
        meta["Title"]   = "Foundation Model Embeddings for Multi-Omics Cancer Classification"
        meta["Author"]  = "Mila Todorovska, Bojana Andonova, Sandra"
        meta["Subject"] = "TCGA Pan-Cancer Classification · Geneformer · MethylGPT · MOFA+"

        page_title(pdf);          print("  ✓ Page 1 — Title")
        page_abstract(pdf);       print("  ✓ Page 2 — Abstract & Introduction")
        page_methods(pdf);        print("  ✓ Page 3 — Methods")
        page_results_table(pdf);  print("  ✓ Page 4 — Results table")
        page_umap(pdf);           print("  ✓ Page 5 — UMAP")
        page_roc_confusion(pdf);  print("  ✓ Page 6 — ROC + Confusion matrix")
        page_shap(pdf);           print("  ✓ Page 7 — SHAP + Per-class")
        page_kegg(pdf);           print("  ✓ Page 8 — KEGG pathways")
        page_discussion(pdf);     print("  ✓ Page 9 — Discussion & Conclusion")

    print(f"\n✓ Saved → {OUTPUT}")
    print(f"  Open with: open {OUTPUT}")

if __name__ == "__main__":
    main()
