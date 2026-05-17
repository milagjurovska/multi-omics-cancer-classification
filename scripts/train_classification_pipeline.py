import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_RNA_EMBEDDINGS = Path("data/processed/geneformer_embeddings.npy")
DEFAULT_RNA_METADATA = Path("data/processed/tcga_rna_seq.h5ad")
DEFAULT_METHYLATION_EMBEDDINGS = Path("data/processed/methylgpt_embeddings.npy")
DEFAULT_METHYLATION_METADATA = Path("data/processed/tcga_methylation.h5ad")
DEFAULT_OUTPUT_DIR = Path("outputs/classification")
DNA_EMBEDDING_CANDIDATES = [
    Path("data/processed/methylgpt_embeddings.npy"),
    Path("data/processed/cpgpt_embeddings.npy"),
    Path("data/processed/dna_methylation_embeddings.npy"),
]


@dataclass
class OmicsDataset:
    name: str
    features: np.ndarray
    metadata: pd.DataFrame


def resolve_embedding_path(primary_path: Path, candidates: List[Path], name: str) -> Path:
    if primary_path.exists():
        return primary_path
    for candidate in candidates:
        if candidate.exists():
            print(f"Using {name} embeddings from {candidate}")
            return candidate
    return primary_path


def load_omics_dataset(
    name: str,
    embeddings_path: Path,
    metadata_path: Path,
    sample_col: str,
    label_col: str,
) -> Optional[OmicsDataset]:
    if not embeddings_path.exists():
        print(f"Skipping {name}: embeddings not found at {embeddings_path}")
        return None
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file for {name} not found: {metadata_path}")

    features = np.load(embeddings_path)
    metadata = ad.read_h5ad(metadata_path).obs.copy().reset_index(drop=True)

    required_cols = {sample_col, label_col}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing_cols)}")
    if len(metadata) != features.shape[0]:
        raise ValueError(
            f"{name} sample mismatch: metadata has {len(metadata)} rows but "
            f"embeddings have {features.shape[0]} rows."
        )

    metadata = metadata[[sample_col, label_col]].copy()
    metadata[sample_col] = metadata[sample_col].astype(str)
    metadata[label_col] = metadata[label_col].astype(str)
    return OmicsDataset(name=name, features=features, metadata=metadata)


def align_modalities(
    left: OmicsDataset,
    right: OmicsDataset,
    sample_col: str,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    left_frame = left.metadata.copy()
    right_frame = right.metadata.copy()
    left_frame["_left_index"] = np.arange(len(left_frame))
    right_frame["_right_index"] = np.arange(len(right_frame))

    merged = left_frame.merge(
        right_frame,
        on=sample_col,
        suffixes=("_left", "_right"),
        how="inner",
    )
    left_label = f"{label_col}_left"
    right_label = f"{label_col}_right"
    label_mismatch = merged[merged[left_label] != merged[right_label]]
    if not label_mismatch.empty:
        raise ValueError(
            "Found samples with different labels between modalities: "
            f"{label_mismatch[sample_col].head().tolist()}"
        )
    if merged.empty:
        raise ValueError("No overlapping samples found between modalities.")

    left_features = left.features[merged["_left_index"].to_numpy()]
    right_features = right.features[merged["_right_index"].to_numpy()]
    labels = merged[left_label].to_numpy()
    aligned_metadata = merged[[sample_col, left_label]].rename(columns={left_label: label_col})
    return left_features, right_features, labels, aligned_metadata


def build_model(model_name: str, random_state: int):
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        alpha=1e-3,
                        early_stopping=True,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it or choose another model."
            ) from exc
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def choose_fold_count(labels: np.ndarray, requested_folds: int) -> int:
    _, counts = np.unique(labels, return_counts=True)
    max_folds = int(counts.min())
    if max_folds < 2:
        raise ValueError("Each cancer type needs at least 2 samples for cross-validation.")
    return min(requested_folds, max_folds)


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_single_view(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    view_name: str,
    folds: int,
    random_state: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    estimator = build_model(model_name, random_state)

    rows = []
    prediction_frames = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y), start=1):
        model = clone(estimator)
        model.fit(features[train_idx], y[train_idx])

        pred = model.predict(features[test_idx])
        proba = model.predict_proba(features[test_idx])
        rows.append(
            {
                "view": view_name,
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
                "f1_weighted": f1_score(y[test_idx], pred, average="weighted"),
                "auc_ovr_macro": safe_auc(y[test_idx], proba),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "view": view_name,
                    "model": model_name,
                    "fold": fold,
                    "row_index": test_idx,
                    "true_label": encoder.inverse_transform(y[test_idx]),
                    "predicted_label": encoder.inverse_transform(pred),
                }
            )
        )

    fold_metrics = pd.DataFrame(rows)
    summary = summarize_fold_metrics(fold_metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return summary, predictions


def evaluate_late_fusion(
    methylation_features: np.ndarray,
    rna_features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    folds: int,
    random_state: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    base_estimator = build_model(model_name, random_state)

    rows = []
    prediction_frames = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(methylation_features, y), start=1):
        methylation_model = clone(base_estimator)
        rna_model = clone(base_estimator)
        methylation_model.fit(methylation_features[train_idx], y[train_idx])
        rna_model.fit(rna_features[train_idx], y[train_idx])

        methylation_proba = methylation_model.predict_proba(methylation_features[test_idx])
        rna_proba = rna_model.predict_proba(rna_features[test_idx])
        fused_proba = (methylation_proba + rna_proba) / 2.0
        pred = fused_proba.argmax(axis=1)

        rows.append(
            {
                "view": "late_fusion",
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
                "f1_weighted": f1_score(y[test_idx], pred, average="weighted"),
                "auc_ovr_macro": safe_auc(y[test_idx], fused_proba),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "view": "late_fusion",
                    "model": model_name,
                    "fold": fold,
                    "row_index": test_idx,
                    "true_label": encoder.inverse_transform(y[test_idx]),
                    "predicted_label": encoder.inverse_transform(pred),
                }
            )
        )

    fold_metrics = pd.DataFrame(rows)
    summary = summarize_fold_metrics(fold_metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return summary, predictions


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> Dict[str, float]:
    first = fold_metrics.iloc[0]
    summary = {"view": first["view"], "model": first["model"], "folds": len(fold_metrics)}
    for metric in ["accuracy", "f1_macro", "f1_weighted", "auc_ovr_macro"]:
        summary[f"{metric}_mean"] = float(fold_metrics[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics[metric].std(ddof=0))
    return summary


def parse_models(values: Iterable[str]) -> List[str]:
    models = []
    for value in values:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    return models


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methylation = load_omics_dataset(
        "dna_methylation",
        resolve_embedding_path(
            args.methylation_embeddings,
            DNA_EMBEDDING_CANDIDATES,
            "DNA methylation",
        ),
        args.methylation_metadata,
        args.sample_col,
        args.label_col,
    )
    rna = load_omics_dataset(
        "rna",
        args.rna_embeddings,
        args.rna_metadata,
        args.sample_col,
        args.label_col,
    )

    if methylation is None and rna is None:
        raise FileNotFoundError("No embeddings were found. Provide at least one modality.")

    summaries = []
    predictions = []
    models = parse_models(args.models)

    if methylation is not None:
        folds = choose_fold_count(methylation.metadata[args.label_col].to_numpy(), args.folds)
        for model_name in models:
            summary, prediction = evaluate_single_view(
                methylation.features,
                methylation.metadata[args.label_col].to_numpy(),
                model_name,
                "dna_methylation",
                folds,
                args.random_state,
            )
            summaries.append(summary)
            predictions.append(prediction)

    if rna is not None:
        folds = choose_fold_count(rna.metadata[args.label_col].to_numpy(), args.folds)
        for model_name in models:
            summary, prediction = evaluate_single_view(
                rna.features,
                rna.metadata[args.label_col].to_numpy(),
                model_name,
                "rna",
                folds,
                args.random_state,
            )
            summaries.append(summary)
            predictions.append(prediction)

    if methylation is not None and rna is not None:
        methylation_x, rna_x, labels, aligned_metadata = align_modalities(
            methylation,
            rna,
            args.sample_col,
            args.label_col,
        )
        aligned_metadata.to_csv(output_dir / "aligned_samples.csv", index=False)
        folds = choose_fold_count(labels, args.folds)
        early_fusion_features = np.concatenate([methylation_x, rna_x], axis=1)
        for model_name in models:
            summary, prediction = evaluate_single_view(
                early_fusion_features,
                labels,
                model_name,
                "early_fusion",
                folds,
                args.random_state,
            )
            summaries.append(summary)
            predictions.append(prediction)

            summary, prediction = evaluate_late_fusion(
                methylation_x,
                rna_x,
                labels,
                model_name,
                folds,
                args.random_state,
            )
            summaries.append(summary)
            predictions.append(prediction)

    metrics = pd.DataFrame(summaries)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved metrics to {output_dir / 'metrics_summary.csv'}")
    print(metrics.sort_values("f1_macro_mean", ascending=False).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate cancer classifiers from foundation-model embeddings."
    )
    parser.add_argument("--rna-embeddings", type=Path, default=DEFAULT_RNA_EMBEDDINGS)
    parser.add_argument("--rna-metadata", type=Path, default=DEFAULT_RNA_METADATA)
    parser.add_argument(
        "--methylation-embeddings",
        "--dna-embeddings",
        dest="methylation_embeddings",
        type=Path,
        default=DEFAULT_METHYLATION_EMBEDDINGS,
        help=(
            "DNA methylation embeddings. Defaults to methylgpt_embeddings.npy, "
            "with automatic fallback to cpgpt_embeddings.npy or dna_methylation_embeddings.npy."
        ),
    )
    parser.add_argument(
        "--methylation-metadata",
        "--dna-metadata",
        dest="methylation_metadata",
        type=Path,
        default=DEFAULT_METHYLATION_METADATA,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-col", default="case_id")
    parser.add_argument("--label-col", default="cancer_type")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost"],
        help="Model names, separated by spaces or commas. Choices: logistic_regression, random_forest, mlp, xgboost.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
