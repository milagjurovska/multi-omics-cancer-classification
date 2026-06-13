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

### Step 1 — Run the main classifier

#### What does `train_classifier.py` actually do?

It runs 4 experiments on your 800 cancer patients and measures how well each approach classifies cancer type:

| Experiment | What it uses | Why |
|---|---|---|
| Geneformer only | RNA embeddings (1152-dim) | Single modality baseline |
| MethylGPT only | Methylation embeddings (128-dim) | Single modality baseline |
| Early Fusion | Both concatenated (1280-dim) | Simple multi-omics combination |
| Late Fusion | Average of both models' predictions | Ensemble approach |

For each experiment it:
- Splits 800 patients into 5 groups (k-fold cross-validation) — trains on 4, tests on 1, repeats 5 times
- Reports **Accuracy**, **F1 Score**, **AUC-ROC** for each fold and as a mean
- Saves plots: UMAP, ROC curves, confusion matrix, SHAP feature importance, per-class metrics

#### How to run it

**First** — download these 2 files from Drive (`data/processed/`) into your local `data/processed/` folder:
- `geneformer_embeddings.npy`
- `methylgpt_embeddings.npy`

**Install dependencies** (first time only):
```bash
cd /Users/sani/Desktop/rudarenje/multi-omics-cancer-classification
pip3 install xgboost shap umap-learn scikit-learn matplotlib pandas numpy anndata
```

**Run the classifier:**
```bash
python3 scripts/train_classifier.py
```

Takes ~15 minutes. When done, the `results/` folder will contain:

| File | What it is |
|---|---|
| `metrics_summary.csv` | Accuracy, F1, AUC for each experiment |
| `per_class_metrics.csv` | Precision, Recall, F1 per cancer type |
| `umap_embeddings.png` | 2D visualization of how cancer types cluster |
| `roc_curves.png` | ROC curve per cancer type for the best model |
| `confusion_matrix.png` | Which cancers get confused with which |
| `shap_summary.png` | Which embedding dimensions matter most |
| `metrics_comparison.png` | Bar chart comparing all experiments |

---

### Step 2 — Add the Autoencoder experiment

Once Bojana sends `autoencoder_embeddings.npy`:
1. Put it in `data/processed/`
2. Run the classifier again — it will automatically pick it up as Experiment 5

```bash
python3 scripts/train_classifier.py
```

---

### Step 3 — Build the final comparison table

Once you have your results and Mila sends the MOFA+ CSV, combine everything into one table:

| Method | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| MOFA+ classical baseline | (from Mila) | | |
| Geneformer only | (from results/) | | |
| MethylGPT only | (from results/) | | |
| Early Fusion (concat) | (from results/) | | |
| Late Fusion (ensemble) | (from results/) | | |
| Deep Fusion (Autoencoder) | (from Bojana) | | |

This table directly answers all 3 research questions from the proposal and is the core result of the paper.

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
