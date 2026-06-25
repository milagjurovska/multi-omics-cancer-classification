# Task Plan — Multi-Omics Cancer Classification

**800 samples · 6 cancer types · 3 people · all start simultaneously**

---

## PERSON 1 — MILA: MOFA+ Baseline

### What is MOFA+ and why do you need it?

Right now you have Geneformer and MethylGPT — two AI foundation models. But the professor will ask: *"Is this actually better than existing methods, or did you just use fancy tools for no reason?"*

MOFA+ (Multi-Omics Factor Analysis) is the **standard classical method** that researchers have used for years to combine methylation + RNA data. It finds shared patterns between the two data types without any AI — just math (matrix factorization).

By running MOFA+ and comparing its accuracy to your foundation model results, you directly answer Research Question 2 from your proposal:
> *"Дали репрезентациите добиени од foundation модели се поинформативни од суровите метилациски податоци?"*

If your foundation models get 95% accuracy and MOFA+ gets 80% → strong result. If they're equal → also an interesting finding worth discussing.

### How to run it

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/mofa_baseline_colab.ipynb` from your computer
3. **No GPU needed** — Runtime → Change runtime type → **None (CPU)**
4. Mount Drive when asked
5. Runtime → Run all
6. Takes ~30 minutes
7. Creates `results/mofa/baseline_metrics.csv` in Drive — a table with MOFA+ accuracy numbers

**When done: send `baseline_metrics.csv` to Sandra.** She adds it to the final comparison table.

---

### Task 2 — KEGG Pathway Analysis (Biological Interpretation)

#### What is KEGG and why does it matter?

After Sandra runs the classifier, the SHAP plot will tell you "embedding dimension 47 is important." That number means nothing biologically — it's just a position in a vector.

KEGG (Kyoto Encyclopedia of Genes and Genomes) is a database that maps every CpG probe in the human genome to known biological pathways — things like "DNA repair", "cell cycle regulation", "PI3K-Akt signaling". These are the actual molecular mechanisms of cancer.

KEGG analysis lets you say something like:
> *"The CpG sites most discriminative between cancer types are located in genes involved in tumor suppressor pathways and Wnt signaling — both well-known cancer drivers."*

This directly answers Research Question 3 from the proposal:
> *"Кои молекуларни карактеристики (CpG локуси, гени) најмногу придонесуваат за разликување на различните типови карциноми?"*

Without it, you have numbers. With it, you have **biological insight** — that is what separates a good project from an excellent one.

#### What KEGG analysis actually does, step by step

1. Load `tcga_methylation.h5ad` — this has beta values for 800 patients × 49156 CpG probes
2. For each probe, run an ANOVA test — measures how different the methylation level is across the 6 cancer types (high F-statistic = this probe strongly distinguishes cancers)
3. Take the top 500 most discriminative probes by F-statistic
4. Download the **Illumina 450K annotation file** (free CSV) — this maps every probe ID (e.g. `cg00000029`) to the gene it sits in (e.g. `TAC1`)
5. Collect the list of gene names for your top 500 probes
6. Run KEGG enrichment with `gseapy` — it compares your gene list against all known KEGG pathways and finds which pathways appear more than expected by chance
7. Plot the top 20 enriched pathways as a horizontal bar chart

#### How to run it

1. Open Google Colab
2. Upload `notebooks/kegg_analysis_colab.ipynb`
3. **No GPU needed** — Runtime → Change runtime type → **None (CPU)**
4. Mount Drive when asked
5. Runtime → Run all
6. Takes ~20 minutes
7. Creates in Drive:
   - `results/kegg_top_pathways.png` — bar chart of top enriched pathways
   - `results/kegg_results.csv` — full enrichment table with p-values
   - `results/top_discriminative_probes.csv` — the 500 most important CpG probes with gene names

**This task is completely independent — does not wait for Sandra or Bojana. Start it right after MOFA+.**

**When done: share `kegg_top_pathways.png` with the group** — this figure goes directly into the final report.

---

## PERSON 2 — BOJANA: Autoencoder (Deep Fusion)

### What is an autoencoder and why do you need it?

You have two types of embeddings:
- **Geneformer**: 1152 numbers per patient, from RNA data
- **MethylGPT**: 128 numbers per patient, from methylation data

Right now Early Fusion just **glues them together** (1152 + 128 = 1280 numbers). This is the simplest possible approach.

An **autoencoder** is a neural network that learns a smarter combination. It takes both sets of numbers, compresses them into **64 shared numbers** (the latent space), and learns to reconstruct both original sets from those 64 numbers. The 64 numbers capture what is shared between RNA and methylation — the deepest biological signal.

This is the third fusion strategy you promised the professor:
> *"Длабока интеграција — автоенкодер за учење на заедничка латентна репрезентација"*

Without this, you are missing something explicitly stated in the proposal.

### How to run it

1. Open Google Colab
2. Upload `notebooks/autoencoder_fusion_colab.ipynb`
3. Runtime → Change runtime type → **T4 GPU** (important — it trains faster)
4. Mount Drive when asked
5. Runtime → Run all
6. Takes ~20 minutes
7. Creates `data/processed/autoencoder_embeddings.npy` in Drive
   - Shape: 800 × 64 — one 64-dimensional representation per patient
   - Also saves a UMAP plot and metrics to Drive

**When done: download `autoencoder_embeddings.npy` and send it to Sandra.**

---

## PERSON 3 — SANDRA: Run Classifier + Combine Everything

### Step 1 — Install dependencies (first time only)

Open Terminal, go to the project folder, and run:

```bash
cd /Users/sani/Desktop/rudarenje/multi-omics-cancer-classification
pip3 install xgboost shap umap-learn scikit-learn matplotlib pandas numpy anndata
```

This installs everything needed. You only do this once.

---

### Step 2 — Download files from Drive

Go to your shared Google Drive folder → `data/processed/`

Download these 2 files and put them in your local `data/processed/` folder:
- `geneformer_embeddings.npy`
- `methylgpt_embeddings.npy`

Your local folder should look like:
```
data/processed/
    geneformer_embeddings.npy   ← download from Drive
    methylgpt_embeddings.npy    ← download from Drive
    probe_ids_type3.csv
    ...
```

---

### Step 3 — Run the classifier

#### What does `train_classifier.py` actually do?

It runs experiments on your 800 cancer patients and measures how well each approach classifies the cancer type:

| Experiment | What it uses | Why |
|---|---|---|
| Geneformer only | RNA embeddings (1152 numbers/patient) | Single modality |
| MethylGPT only | Methylation embeddings (128 numbers/patient) | Single modality |
| Early Fusion | Both glued together (1280 numbers/patient) | Simple combination |
| Late Fusion | Average of both models' predictions | Ensemble |
| Deep Fusion | Autoencoder latent space (64 numbers/patient) | Learned combination — added automatically when Bojana's file arrives |

For each experiment it:
- Splits 800 patients into 5 groups — trains on 4 groups, tests on 1, repeats 5 times (k-fold cross-validation)
- Reports **Accuracy**, **F1 Score**, **AUC-ROC** with mean and standard deviation
- Saves all plots automatically

**Run it:**
```bash
python3 scripts/train_classifier.py
```

Takes ~15 minutes. When done, `results/` folder contains:

| File | What it is |
|---|---|
| `metrics_summary.csv` | Main results table — Accuracy, F1, AUC per experiment |
| `per_class_metrics.csv` | Precision, Recall, F1 broken down per cancer type |
| `umap_embeddings.png` | 2D map showing how the 800 patients cluster by cancer |
| `roc_curves.png` | ROC curves per cancer type for the best model |
| `confusion_matrix.png` | Which cancers get confused with which |
| `shap_summary.png` | Which embedding dimensions matter most for predictions |
| `metrics_comparison.png` | Bar chart comparing all experiments side by side |

---

### Step 4 — Add Autoencoder (after Bojana finishes, ~20 min wait)

When Bojana sends you `autoencoder_embeddings.npy`:
1. Put it in `data/processed/`
2. Run the classifier again — it automatically detects the new file and adds it as Experiment 5

```bash
python3 scripts/train_classifier.py
```

---

### Step 5 — Build the final comparison table (after Mila and Bojana finish)

When Mila sends you `baseline_metrics.csv` from her MOFA+ run:
1. Create folder `results/mofa/` if it doesn't exist
2. Put `baseline_metrics.csv` inside it
3. Run:

```bash
python3 scripts/combine_results.py
```

This automatically:
- Loads your results from `results/metrics_summary.csv`
- Loads Mila's MOFA+ results from `results/mofa/baseline_metrics.csv`
- Combines everything into one table
- Saves `results/final_comparison.csv` and `results/final_comparison.png`

The final table will look like:

| Method | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| MOFA+ (classical baseline) | from Mila | | |
| Geneformer only | from your run | | |
| MethylGPT only | from your run | | |
| Early Fusion | from your run | | |
| Late Fusion | from your run | | |
| Deep Fusion (Autoencoder) | from your run | | |

**This is the core result of the entire project.** Share `final_comparison.png` with the group.

---

## Order of Operations

```
Mila   ──► run MOFA+ (~30 min) ──► run KEGG (~20 min) ────────► send CSV + PNG to Sandra
Bojana ──► run Autoencoder (~20 min) ───────────────────────────► send .npy to Sandra
Sandra ──► run classifier (~15 min) ──► add AE ──► final table ──► DONE
```

**All three start right now. No waiting on each other for the first step.**

- Sandra's Step 2 waits for Bojana (~20 min)
- Sandra's Step 3 waits for both Mila and Bojana
- Mila's KEGG does not wait for anyone — runs independently

---

## What the final project answers

| Research Question (from proposal) | Answered by |
|---|---|
| Does multi-omics beat single-modality? | Early / Late / Deep Fusion vs Geneformer only / MethylGPT only |
| Are foundation models better than raw data? | Foundation models vs MOFA+ baseline (Mila) |
| Which features drive cancer classification? | KEGG pathway analysis (Mila) + SHAP plot (Sandra) |
