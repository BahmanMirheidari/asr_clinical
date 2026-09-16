"""
Experiment Results Aggregator - Enhanced Version
Supports BOTH classification and regression experiments with subgroup analysis (Dys/Norm).

Regression ranking metric: RMSE (lower is better).
Classification ranking metric: F1-Macro (higher is better).

Baselines (renamed):
  audio_only  -> 'Clinical-Feature-Only'
  text_only   -> 'Text-Embedding-Only'

Prediction file discovery (two-phase):
  Phase 1 — fusion_results/leakage_safe_5fold/
            <method>_oof_predictions.csv  (primary source for ALL methods)
  Phase 2 — fusion_results/meta_fusion/  (fallback, only if a key was not
            already filled by Phase 1)

Metric sourcing:
  For every (model, method) pair the aggregator builds a row from BOTH
  sources and merges them:
    * JSON aggregate metrics (when available) take precedence.
    * Any missing metric is computed directly from the prediction CSV —
      including ALL classification metrics (sensitivity, specificity,
      macro-F1, balanced accuracy, ROC-AUC) and their Dys/Typ subgroup
      variants, plus regression rmse / r2. This guarantees that ensemble_*
      predictions (ensemble_weighted, ensemble_voting, ...) always appear
      in the final dataframe even when no matching meta_fusion_metrics.json
      entry exists.

Plot-data CSVs (all_results.csv, summary_table_flat_*.csv,
sens_spec_three_marker_data_*.csv) are consequently populated for every
discovered method, including ensembles.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, linregress, pearsonr
from scipy.stats import rankdata
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')
        sns.set_style("whitegrid")

sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# =======================================================================
#  GLOBAL ENSEMBLE / META-FUSION HELPERS
# =======================================================================

ENSEMBLE_IDENTIFIERS: Set[str] = {
    'ensemble_average', 'ensemble_voting', 'ensemble_stacking',
    'ensemble_weighted', 'ensemble_confidence_selection', 'ensemble_best',
    'meta_fusion', 'meta_fusion_summary',
    'Average Ensemble', 'Voting Ensemble', 'Stacking Ensemble',
    'Weighted Ensemble', 'Confidence Selection', 'Best Method',
    'Meta-Fusion', 'Meta Fusion', 'Meta-Fusion Ensemble',
}


def is_ensemble_method(method_key: Any) -> bool:
    """True if method_key refers to any ensemble / meta-fusion method."""
    if method_key is None or not isinstance(method_key, str):
        return False
    if method_key in ENSEMBLE_IDENTIFIERS:
        return True
    ml = method_key.lower().strip()
    if ml.startswith('ensemble_') or 'ensemble' in ml:
        return True
    stripped = ml.replace('-', '').replace('_', '').replace(' ', '')
    if stripped in ('metafusion', 'metafusionsummary'):
        return True
    return False


def filter_out_ensembles(df: pd.DataFrame, method_col: str = 'Method',
                         label_col: str = 'Method_Label') -> pd.DataFrame:
    """Return a copy of df with all ensemble / meta-fusion rows removed."""
    if df.empty:
        return df.copy()
    mask = pd.Series(False, index=df.index)
    if method_col in df.columns:
        mask = mask | df[method_col].apply(is_ensemble_method)
    if label_col in df.columns:
        mask = mask | df[label_col].apply(is_ensemble_method)
    return df[~mask].copy()


def is_clinical_feature_only(method_key: Any) -> bool:
    """True for the clinical-feature-only baseline (formerly audio-only)."""
    if not isinstance(method_key, str):
        return False
    ml = method_key.lower().strip()
    if ml in ('audio_only', 'audio-only', 'audio only'):
        return True
    cf = ml.replace('-', ' ').replace('_', ' ').strip()
    return cf in ('clinical feature only', 'clinical-feature-only')


def is_text_embedding_only(method_key: Any) -> bool:
    """True for the text-embedding-only baseline (formerly text-only)."""
    if not isinstance(method_key, str):
        return False
    ml = method_key.lower().strip()
    if ml in ('text_only', 'text-only', 'text only'):
        return True
    te = ml.replace('-', ' ').replace('_', ' ').strip()
    return te in ('text embedding only', 'text-embedding-only')


def is_baseline_method(method_key: Any) -> bool:
    """True for any unimodal baseline (clinical-feature-only or text-embedding-only)."""
    return is_clinical_feature_only(method_key) or is_text_embedding_only(method_key)


def is_plausible_regression_metric(value: Any, metric_name: str) -> bool:
    """Plausibility filter for regression metrics."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    if metric_name == 'rmse':
        return v > 1e-9
    if metric_name == 'r2':
        return -2.0 < v <= 1.0 + 1e-9
    return True


def strip_predictions_suffix(stem: str) -> str:
    """Strip '_oof_predictions' or '_predictions' from a file stem."""
    s = stem
    for suffix in ('_oof_predictions', '_predictions'):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return s


def canonical_ensemble_key(method_part: str) -> Optional[str]:
    """Map a meta_fusion prediction stem to the canonical ensemble_* key."""
    if not isinstance(method_part, str):
        return None
    mp = method_part.strip()
    if mp in ('meta_fusion', 'meta_fusion_summary'):
        return 'meta_fusion'
    if mp.startswith('ensemble_'):
        return mp
    known_strategies = {'average', 'voting', 'stacking', 'weighted',
                        'confidence_selection', 'best'}
    if mp in known_strategies:
        return f'ensemble_{mp}'
    return None


# =======================================================================
#  CONFIGURATION
# =======================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the aggregator.

    Ranking metrics:
      - classification: F1-Macro (higher is better)
      - regression:     RMSE      (lower is better)

    Baselines (renamed):
      audio_only  -> 'Clinical-Feature-Only'
      text_only   -> 'Text-Embedding-Only'
    """
    fusion_methods: Dict[str, str] = field(default_factory=lambda: {
        'audio_only': 'audio_only_aggregate_metrics.json',
        'text_only': 'text_only_aggregate_metrics.json',
        'early': 'early_aggregate_metrics.json',
        'late': 'late_aggregate_metrics.json',
        'confidence': 'confidence_aggregate_metrics.json',
        'interaction': 'interaction_aggregate_metrics.json',
        'moe': 'moe_aggregate_metrics.json',
        'mlp': 'mlp_aggregate_metrics.json',
        'stacking': 'stacking_aggregate_metrics.json',
        'cca': 'cca_aggregate_metrics.json',
        'dynamic': 'dynamic_aggregate_metrics.json',
        'cross_attention': 'cross_attention_aggregate_metrics.json',
        'adaptive_weighted': 'adaptive_weighted_aggregate_metrics.json',
        'bilinear': 'bilinear_aggregate_metrics.json'
    })

    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        'audio_only': 'Clinical-Feature-Only',
        'text_only': 'Text-Embedding-Only',
        'early': 'Early Fusion',
        'late': 'Late Fusion',
        'confidence': 'Confidence-Weighted',
        'interaction': 'Interaction Stacking',
        'moe': 'Mixture of Experts',
        'mlp': 'MLP Fusion',
        'stacking': 'Model-Based Stacking',
        'cca': 'CCA Fusion',
        'dynamic': 'Dynamic Fusion',
        'cross_attention': 'Cross-Attention Fusion',
        'adaptive_weighted': 'Adaptive Weighted Fusion',
        'bilinear': 'Bilinear Fusion'
    })

    fusion_combination_display_names: Dict[str, str] = field(default_factory=lambda: {
        'fuse-audio_text': 'Clinical + Text',
        'fuse-audio_early': 'Clinical + Early',
        'fuse-text_early': 'Text + Early',
        'fuse-audio_text_early': 'Clinical + Text + Early',
        'fuse-early_late': 'Early + Late',
        'fuse-audio_text_late': 'Clinical + Text + Late',
        'fuse-confidence_audio': 'Confidence + Clinical',
        'fuse-confidence_text': 'Confidence + Text',
        'fuse-confidence_early': 'Confidence + Early',
        'fuse-moe_stacking': 'MoE + Stacking',
        'fuse-cca_mlp': 'CCA + MLP',
        'fuse-dynamic_confidence': 'Dynamic + Confidence',
        'fuse-cross_attention_bilinear': 'Cross-Attention + Bilinear',
        'fuse-adaptive_weighted_confidence': 'Adaptive + Confidence',
        'fuse-mlp_cross_attention': 'MLP + Cross-Attention',
    })

    meta_fusion_display_names: Dict[str, str] = field(default_factory=lambda: {
        'average': 'Average Ensemble',
        'voting': 'Voting Ensemble',
        'stacking': 'Stacking Ensemble',
        'weighted': 'Weighted Ensemble',
        'confidence_selection': 'Confidence Selection',
        'best': 'Best Method'
    })

    model_name_mapping: Dict[str, str] = field(default_factory=lambda: {
        'microsoft.deberta-v3-base': 'DeBERTa-Base',
        'microsoft.deberta-v3-large': 'DeBERTa-Large',
        'distilroberta-base': 'DistilRoBERTa',
        'roberta-base': 'RoBERTa-Base',
        'roberta-large': 'RoBERTa-Large',
        'bert-base-uncased': 'BERT-Base',
        'bert-large-uncased': 'BERT-Large',
        'albert-base-v2': 'ALBERT-Base',
        'albert-large-v2': 'ALBERT-Large',
        'clinicalbert': 'ClinicalBERT',
        'biomed-roberta-base': 'BioMed-RoBERTa',
        'scibert-scivocab-uncased': 'SciBERT',
        'legal-bert-base': 'LegalBERT',
        'pubmedbert-base': 'PubMedBERT',
        'bioclinicalbert': 'BioClinicalBERT',
    })

    classification_metrics: List[str] = field(default_factory=lambda: [
        'macro_f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'roc_auc'
    ])

    regression_metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'r2'
    ])

    classification_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_macro_f1', 'subgroup_sensitivity', 'subgroup_specificity',
        'subgroup_balanced_accuracy', 'subgroup_roc_auc',
        'non_subgroup_macro_f1', 'non_subgroup_sensitivity',
        'non_subgroup_specificity', 'non_subgroup_balanced_accuracy',
        'non_subgroup_roc_auc'
    ])

    regression_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_rmse', 'subgroup_r2',
        'non_subgroup_rmse', 'non_subgroup_r2'
    ])

    metric_labels: Dict[str, str] = field(default_factory=lambda: {
        'macro_f1': 'F1-Macro',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'balanced_accuracy': 'Balanced Accuracy',
        'roc_auc': 'AUC-ROC',
        'rmse': 'RMSE',
        'r2': 'R²',
        'subgroup_macro_f1': 'Dys - F1-Macro',
        'subgroup_sensitivity': 'Dys - Sensitivity',
        'subgroup_specificity': 'Dys - Specificity',
        'subgroup_balanced_accuracy': 'Dys - Balanced Accuracy',
        'subgroup_roc_auc': 'Dys - AUC-ROC',
        'non_subgroup_macro_f1': 'Typ - F1-Macro',
        'non_subgroup_sensitivity': 'Typ - Sensitivity',
        'non_subgroup_specificity': 'Typ - Specificity',
        'non_subgroup_balanced_accuracy': 'Typ - Balanced Accuracy',
        'non_subgroup_roc_auc': 'Typ - AUC-ROC',
        'subgroup_rmse': 'Dys - RMSE',
        'subgroup_r2': 'Dys - R²',
        'non_subgroup_rmse': 'Typ - RMSE',
        'non_subgroup_r2': 'Typ - R²'
    })

    ranking_metric_classification: str = 'macro_f1'
    ranking_metric_regression: str = 'rmse'

    bootstrap_iterations: int = 1000

    fusion_method_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'Base Methods': ['audio_only', 'text_only', 'early', 'late'],
        'Advanced Methods': ['confidence', 'interaction', 'moe', 'mlp',
                             'stacking', 'cca', 'dynamic'],
        'State-of-the-Art': ['cross_attention', 'adaptive_weighted', 'bilinear']
    })


# =======================================================================
#  HELPER FUNCTIONS
# =======================================================================

def get_method_display_name(method_key: str, config: ExperimentConfig) -> str:
    if method_key in config.method_display_names:
        return config.method_display_names[method_key]

    if isinstance(method_key, str) and method_key.startswith('fuse-'):
        if method_key in config.fusion_combination_display_names:
            return config.fusion_combination_display_names[method_key]
        parts = method_key.replace('fuse-', '').split('_')
        return ' + '.join([p.replace('_', ' ').title() for p in parts])

    if isinstance(method_key, str) and method_key.startswith('ensemble_'):
        ensemble = method_key.replace('ensemble_', '')
        return config.meta_fusion_display_names.get(ensemble, ensemble.title())

    if method_key == 'meta_fusion':
        return 'Meta-Fusion'

    if not isinstance(method_key, str):
        return str(method_key)
    return method_key.replace('_', ' ').title()


def clean_model_name(model_name: str) -> str:
    cleaned = model_name
    prefixes_to_remove = [
        'ecas_105-', 'ecas105-', 'ecas_105_', 'ecas105_',
        'ecas-105-', 'ecas-105_',
    ]
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    if re.match(r'^\d+-', cleaned):
        cleaned = re.sub(r'^\d+-', '', cleaned)
    cleaned = cleaned.lstrip('-_')
    return cleaned


def detect_model_size(model_name: str) -> str:
    ll = model_name.lower()
    if 'large' in ll:
        return 'Large'
    elif 'base' in ll:
        return 'Base'
    elif 'small' in ll:
        return 'Small'
    return 'Unknown'


def detect_model_family(model_name: str) -> str:
    ll = model_name.lower()
    if 'roberta' in ll:
        return 'RoBERTa'
    elif 'bert' in ll:
        return 'BERT'
    elif 'distil' in ll:
        return 'DistilRoBERTa'
    elif 'albert' in ll:
        return 'ALBERT'
    elif 'deberta' in ll:
        return 'DeBERTa'
    elif 'clinical' in ll:
        return 'Clinical BERT'
    elif 'biomed' in ll:
        return 'BioMed'
    elif 'scibert' in ll:
        return 'SciBERT'
    elif 'legal' in ll:
        return 'Legal BERT'
    return 'Other'


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    result = {'task': 'unknown', 'model': folder_name, 'full_name': folder_name,
              'size': 'Unknown', 'family': 'Other'}

    if 'classification' in folder_name:
        result['task'] = 'classification'
        if folder_name.startswith('classification-fusion-'):
            model_part = folder_name[len('classification-fusion-'):]
        elif folder_name.startswith('classification-'):
            model_part = folder_name[len('classification-'):]
        else:
            model_part = folder_name
    elif 'regression' in folder_name:
        result['task'] = 'regression'
        if folder_name.startswith('regression-fusion-'):
            model_part = folder_name[len('regression-fusion-'):]
        elif folder_name.startswith('regression-'):
            model_part = folder_name[len('regression-'):]
        else:
            model_part = folder_name
    else:
        model_part = folder_name

    model_part = model_part.lstrip('-_')
    model_part = clean_model_name(model_part)
    result['model'] = model_part
    result['size'] = detect_model_size(result['model'])
    result['family'] = detect_model_family(result['model'])
    return result


# =======================================================================
#  SUBGROUP MAPPING
# =======================================================================

def load_dys_speaker_ids(mapping_file: Path) -> List[str]:
    if not mapping_file or not mapping_file.exists():
        print(f"Warning: Dys speaker ID file not found: {mapping_file}")
        return []
    try:
        with open(mapping_file, 'r') as f:
            speaker_ids = [line.strip() for line in f.readlines() if line.strip()]
        print(f"\n✓ Loaded {len(speaker_ids)} Dys speaker IDs from: {mapping_file}")
        if len(speaker_ids) <= 10:
            print(f"  IDs: {speaker_ids}")
        else:
            print(f"  First 5 IDs: {speaker_ids[:5]}")
            print(f"  Last 5 IDs: {speaker_ids[-5:]}")
        return speaker_ids
    except Exception as e:
        print(f"Error loading Dys speaker IDs: {e}")
        return []


# =======================================================================
#  METRIC LOADING
# =======================================================================

def load_metrics_file(file_path: Path) -> Optional[Dict]:
    if not file_path or not file_path.exists():
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_from_result(result: Dict, task: str) -> Dict:
    extracted = {}
    if not isinstance(result, dict):
        return extracted

    if 'all' in result and isinstance(result['all'], dict):
        for key, value in result['all'].items():
            if isinstance(value, (int, float)):
                extracted[key] = value
    else:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in ['threshold', 'threshold_used', 'k_neighbors',
                               'avg_best_k']:
                    extracted[key] = value

    if 'subgroup' in result and isinstance(result['subgroup'], dict):
        for key, value in result['subgroup'].items():
            if isinstance(value, (int, float)):
                extracted[f'subgroup_{key}'] = value

    if 'non_subgroup' in result and isinstance(result['non_subgroup'], dict):
        for key, value in result['non_subgroup'].items():
            if isinstance(value, (int, float)):
                extracted[f'non_subgroup_{key}'] = value

    if 'all' in result and isinstance(result['all'], dict):
        all_data = result['all']
        if 'subgroup' in all_data and isinstance(all_data['subgroup'], dict):
            for key, value in all_data['subgroup'].items():
                if isinstance(value, (int, float)):
                    extracted[f'subgroup_{key}'] = value
        if 'non_subgroup' in all_data and isinstance(all_data['non_subgroup'], dict):
            for key, value in all_data['non_subgroup'].items():
                if isinstance(value, (int, float)):
                    extracted[f'non_subgroup_{key}'] = value
    return extracted


def load_predictions_file(file_path: Path) -> Optional[pd.DataFrame]:
    if not file_path or not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


# =======================================================================
#  PREDICTION-FILE METRIC COMPUTATION  (JSON-independent)
# =======================================================================

def _detect_prediction_columns(pred_df: pd.DataFrame):
    """
    Return (obs_col, pred_col, id_col, prob_col).

    obs_col : ground-truth column (observed / true / target / y_true / ...)
    pred_col: hard label / continuous prediction column
              (predicted / pred / y_pred / output / ...)
    id_col  : speaker / participant / subject / patient / id column, used
              for the Dys vs Typ split.
    prob_col: probability / score column (may equal pred_col when the
              prediction is already a probability).
    """
    cols = list(pred_df.columns)

    def _find_by_exact(exact_set, exclude=None):
        for c in cols:
            if c == exclude:
                continue
            if c.lower().strip() in exact_set:
                return c
        return None

    def _find_by_substring(substrings, exclude=None, require_numeric=False):
        for c in cols:
            if c == exclude:
                continue
            cl = c.lower()
            for s in substrings:
                if s in cl:
                    if require_numeric:
                        v = pd.to_numeric(pred_df[c], errors='coerce')
                        if v.notna().sum() == 0:
                            continue
                    return c
        return None

    obs_col = _find_by_exact(
        {'observed', 'y_true', 'true', 'target', 'label', 'true_label',
         'actual', 'ground_truth', 'observed_label', 'y'}
    )
    if obs_col is None:
        obs_col = _find_by_substring(
            ['observed', 'true', 'actual', 'target', 'y_true', 'label']
        )

    pred_col = _find_by_exact(
        {'predicted', 'y_pred', 'pred', 'prediction', 'output', 'pred_label',
         'predicted_label'},
        exclude=obs_col,
    )
    if pred_col is None:
        pred_col = _find_by_substring(
            ['predicted', 'y_pred', 'pred', 'output'],
            exclude=obs_col, require_numeric=True,
        )

    id_col = _find_by_exact(
        {'speaker_id', 'participant_id', 'subject_id', 'patient_id',
         'speaker', 'participant', 'subject', 'patient', 'id'}
    )
    if id_col is None:
        id_col = _find_by_substring(
            ['speaker', 'participant', 'subject', 'patient', 'id'],
            exclude=obs_col,
        )

    prob_col = _find_by_exact(
        {'probability', 'y_prob', 'prob', 'score', 'proba', 'pred_prob',
         'probabilities', 'y_probability', 'predicted_probability'}
    )
    if prob_col is None:
        prob_col = _find_by_substring(
            ['probability', 'prob', 'y_prob', 'proba'],
            exclude=obs_col,
        )

    # If pred_col is already a probability vector, treat it as prob_col too.
    if prob_col is None and pred_col is not None:
        v = pd.to_numeric(pred_df[pred_col], errors='coerce').dropna()
        if len(v) > 0 and v.between(0, 1).all() and not v.isin([0, 1]).all():
            prob_col = pred_col

    return obs_col, pred_col, id_col, prob_col

def _load_base_ids(folder_path: Path,
                   y_true: np.ndarray) -> Optional[np.ndarray]:
    """
    Find a base-method (non-ensemble) OOF prediction CSV in this experiment
    folder that carries a speaker-ID column, and return its ID vector
    aligned with the given y_true sequence.

    Alignment is verified via exact y_true equality (NaN-safe). Returns
    None if no candidate aligns.
    """
    if folder_path is None:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_true_sig = np.nan_to_num(y_true, nan=-999.0)

    for sub in ('leakage_safe_5fold', 'meta_fusion'):
        d = folder_path / 'fusion_results' / sub
        if not d.exists():
            continue
        for f in sorted(d.glob('*_oof_predictions.csv')):
            stem = strip_predictions_suffix(f.stem)
            if is_ensemble_method(stem):
                continue
            base_df = load_predictions_file(f)
            if base_df is None or base_df.empty:
                continue
            if len(base_df) != len(y_true):
                continue
            b_obs, _, b_id, _ = _detect_prediction_columns(base_df)
            if b_obs is None or b_id is None:
                continue
            b_true = pd.to_numeric(base_df[b_obs], errors='coerce').values
            b_true_sig = np.nan_to_num(b_true, nan=-999.0)
            if not np.array_equal(b_true_sig, y_true_sig):
                continue
            return base_df[b_id].astype(str).values
    return None


def _inject_speaker_ids(pred_df: pd.DataFrame,
                        folder_path: Path) -> pd.DataFrame:
    """
    If pred_df lacks a speaker-ID column, inject one by aligning it with a
    base-method OOF file in the same experiment folder (matched on y_true).
    Ensemble OOF files (e.g. ensemble_confidence_selection_oof_predictions.csv)
    fall into this category — they were written without an ID column.

    Returns a new DataFrame; the input is not mutated.
    """
    obs_col, _, id_col, _ = _detect_prediction_columns(pred_df)
    if id_col is not None:
        return pred_df                      # already has IDs — leave alone
    if obs_col is None:
        return pred_df                      # can't align without y_true

    y_true = pd.to_numeric(pred_df[obs_col], errors='coerce').values
    ids = _load_base_ids(folder_path, y_true)
    if ids is None:
        return pred_df

    out = pred_df.copy()
    out['speaker_id'] = ids
    return out

def _compute_auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUC-ROC using the Mann-Whitney U rank formula (no sklearn).
    Robust to ties. Returns np.nan if a class is missing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_score)
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        prefix: str = '') -> Dict[str, float]:
    """Compute rmse / r2 (with the given prefix) from paired arrays."""
    out: Dict[str, float] = {}
    if len(y_true) < 3:
        return out
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    out[f'{prefix}rmse'] = rmse
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot > 0:
        out[f'{prefix}r2'] = 1.0 - ss_res / ss_tot
    return out


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    prefix: str = '',
) -> Dict[str, float]:
    """
    Compute sensitivity / specificity / macro-F1 / balanced-accuracy / AUC
    (with the given prefix) from hard labels + optional probabilities.
    Positive class is assumed to be 1.
    """
    out: Dict[str, float] = {}
    n = len(y_true)
    if n < 2:
        return out

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    sens_neg = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    f1_pos = (2 * prec_pos * sens / (prec_pos + sens)
              if (prec_pos and sens and prec_pos + sens > 0) else np.nan)
    f1_neg = (2 * prec_neg * sens_neg / (prec_neg + sens_neg)
              if (prec_neg and sens_neg and prec_neg + sens_neg > 0) else np.nan)

    if not np.isnan(sens):
        out[f'{prefix}sensitivity'] = float(sens)
    if not np.isnan(spec):
        out[f'{prefix}specificity'] = float(spec)
    if not (np.isnan(f1_pos) or np.isnan(f1_neg)):
        out[f'{prefix}macro_f1'] = float((f1_pos + f1_neg) / 2.0)
    if not (np.isnan(sens) or np.isnan(spec)):
        out[f'{prefix}balanced_accuracy'] = float((sens + spec) / 2.0)

    if y_prob is not None and len(y_prob) == n:
        auc = _compute_auc_rank(y_true, y_prob)
        if not np.isnan(auc):
            out[f'{prefix}roc_auc'] = auc
    return out


def compute_metrics_from_predictions(
    pred_df: pd.DataFrame,
    task: str,
    dys_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the FULL metric set (combined + subgroup + non_subgroup)
    directly from a prediction DataFrame.

    Classification output keys
    --------------------------
        sensitivity, specificity, macro_f1, balanced_accuracy, roc_auc
        subgroup_*       (Dys split)
        non_subgroup_*   (Typ split)

    Regression output keys
    ----------------------
        rmse, r2
        subgroup_rmse, subgroup_r2
        non_subgroup_rmse, non_subgroup_r2
    """
    out: Dict[str, float] = {}
    obs_col, pred_col, id_col, prob_col = _detect_prediction_columns(pred_df)
    if obs_col is None or pred_col is None:
        return out

    y_true_raw = pd.to_numeric(pred_df[obs_col], errors='coerce')
    y_pred_raw = pd.to_numeric(pred_df[pred_col], errors='coerce')
    valid = y_true_raw.notna() & y_pred_raw.notna()
    n_valid = int(valid.sum())
    if n_valid < 3:
        return out

    y_true_v = y_true_raw[valid].values.astype(float)
    y_pred_v = y_pred_raw[valid].values.astype(float)

    prob_v: Optional[np.ndarray] = None
    if prob_col is not None and prob_col != obs_col:
        prob_v = pd.to_numeric(pred_df[prob_col], errors='coerce')[valid].values

    ids_v: Optional[np.ndarray] = None
    if id_col is not None:
        ids_v = pred_df.loc[valid, id_col].astype(str).values

    if task == 'regression':
        out.update(_regression_metrics(y_true_v, y_pred_v, ''))
        if ids_v is not None and dys_ids:
            dys_set = {str(x) for x in dys_ids}
            dys_mask = np.isin(ids_v, list(dys_set))
            if dys_mask.any():
                out.update(_regression_metrics(
                    y_true_v[dys_mask], y_pred_v[dys_mask], 'subgroup_'))
            if (~dys_mask).any():
                out.update(_regression_metrics(
                    y_true_v[~dys_mask], y_pred_v[~dys_mask], 'non_subgroup_'))
        return out

    # ---- classification ----
    y_true_int = np.round(y_true_v).astype(int)

    # decide if predictions are hard labels or probabilities
    pr = y_pred_v
    is_prob = (len(pr) > 0
               and np.all((pr >= 0) & (pr <= 1))
               and not np.all((pr == 0) | (pr == 1)))
    if is_prob:
        y_pred_int = (pr >= 0.5).astype(int)
        if prob_v is None:
            prob_v = pr
    else:
        y_pred_int = np.round(pr).astype(int)

    out.update(_classification_metrics(y_true_int, y_pred_int, prob_v, ''))

    if ids_v is not None and dys_ids:
        dys_set = {str(x) for x in dys_ids}
        dys_mask = np.isin(ids_v, list(dys_set))
        if dys_mask.any():
            p_d = prob_v[dys_mask] if prob_v is not None else None
            out.update(_classification_metrics(
                y_true_int[dys_mask], y_pred_int[dys_mask], p_d, 'subgroup_'))
        if (~dys_mask).any():
            p_n = prob_v[~dys_mask] if prob_v is not None else None
            out.update(_classification_metrics(
                y_true_int[~dys_mask], y_pred_int[~dys_mask], p_n,
                'non_subgroup_'))
    return out


def load_predictions_unfiltered(experiments: Dict, model_name: str,
                                method_key: str) -> Optional[pd.DataFrame]:
    """Load predictions for a model/method from the discovery cache or disk."""
    if model_name not in experiments:
        return None
    model_data = experiments[model_name]

    pred_key = f'predictions_{method_key}'
    if pred_key in model_data:
        pred_file = model_data[pred_key].get('predictions_file')
        if pred_file and Path(pred_file).exists():
            df = load_predictions_file(Path(pred_file))
            if df is not None:
                df = _inject_speaker_ids(df, model_data[pred_key].get('dir'))
            return df

    for source_key, subdir in [('audio_only', 'leakage_safe_5fold'),
                               ('meta_fusion', 'meta_fusion')]:
        base = model_data.get(source_key, {}).get('dir', '')
        if not base:
            continue
        d = Path(base) / 'fusion_results' / subdir
        if not d.exists():
            continue
        for pattern in (f"{method_key}_oof_predictions.csv",
                        f"{method_key}_predictions.csv",
                        f"{method_key}_oof_predictions.txt",
                        f"{method_key}_predictions.txt",
                        f"*{method_key}*_oof_predictions.csv",
                        f"*{method_key}*_predictions.csv",
                        f"*{method_key}*_predictions.txt"):
            for f in sorted(d.glob(pattern)):
                df = load_predictions_file(f)
                if df is not None:
                    return df
    return None


# =======================================================================
#  EXPERIMENT DISCOVERY
# =======================================================================

PREDICTION_PATTERNS = (
    "*_oof_predictions.csv",
    "*_predictions.csv",
    "*_oof_predictions.txt",
    "*_predictions.txt",
)


def _scan_subdir_for_predictions(subdir: Path, source_name: str,
                                 already_have: Set[str],
                                 overwrite: bool = False) -> Dict[str, Path]:
    """Scan a sub-directory for prediction files; return canonical_key → path."""
    found: Dict[str, Path] = {}
    if not subdir.exists():
        return found

    files = set()
    for pattern in PREDICTION_PATTERNS:
        files.update(subdir.glob(pattern))
    files = sorted(files)

    for pred_file in files:
        method_part = strip_predictions_suffix(pred_file.stem)

        if source_name == 'meta_fusion':
            canonical = canonical_ensemble_key(method_part)
            if canonical is None:
                continue
        else:
            canonical = method_part if method_part else None
            if canonical is None:
                continue

        if canonical in already_have and not overwrite:
            continue
        if canonical in found:
            continue

        found[canonical] = pred_file

    return found


def discover_predictions_for_model(folder_path: Path,
                                   model_name: str, task: str,
                                   size: str, family: str,
                                   experiments: Dict) -> Tuple[int, int]:
    """Two-phase prediction discovery for one model folder."""
    leakage_dir = folder_path / 'fusion_results' / 'leakage_safe_5fold'
    meta_dir = folder_path / 'fusion_results' / 'meta_fusion'

    print(f"    [Phase 1] scanning {leakage_dir.name}/")

    phase1 = _scan_subdir_for_predictions(
        leakage_dir, 'leakage_safe_5fold', already_have=set(), overwrite=False)

    for canonical, pred_file in sorted(phase1.items()):
        key = f'predictions_{canonical}'
        experiments[model_name][key] = {
            'task': task, 'model': model_name, 'size': size,
            'family': family, 'predictions_file': pred_file,
            'dir': folder_path, 'source': 'leakage_safe_5fold',
            'method_key': canonical,
        }
        print(f"      ✓ {pred_file.name}  →  key='{key}'")

    already_have = set(phase1.keys())

    print(f"    [Phase 2] scanning {meta_dir.name}/ (fallback only)")
    phase2 = _scan_subdir_for_predictions(
        meta_dir, 'meta_fusion', already_have=already_have, overwrite=False)

    for canonical, pred_file in sorted(phase2.items()):
        key = f'predictions_{canonical}'
        experiments[model_name][key] = {
            'task': task, 'model': model_name, 'size': size,
            'family': family, 'predictions_file': pred_file,
            'dir': folder_path, 'source': 'meta_fusion',
            'method_key': canonical,
        }
        print(f"      ✓ {pred_file.name}  →  key='{key}'  (fallback)")

    if not phase2:
        print(f"      (nothing new to register)")

    return len(phase1), len(phase2)


def discover_experiments(base_dir: Path, task_type: str = 'all',
                         config: ExperimentConfig = None) -> Dict:
    if config is None:
        config = ExperimentConfig()

    experiments = defaultdict(lambda: defaultdict(dict))

    print(f"\n{'='*70}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*70}")
    print(f"Task type filter: {task_type}")

    all_folders = []
    if task_type in ['classification', 'all']:
        cf = list(base_dir.glob("classification*"))
        all_folders.extend(cf)
        print(f"Found {len(cf)} classification folders")
    if task_type in ['regression', 'all']:
        rf = list(base_dir.glob("regression*"))
        all_folders.extend(rf)
        print(f"Found {len(rf)} regression folders")
    all_folders = sorted(set(all_folders))
    print(f"Total: {len(all_folders)} experiment folders to process")

    for folder_path in all_folders:
        folder_name = folder_path.name
        parsed = parse_folder_name(folder_name)
        model_name = parsed['model']
        task = parsed['task']
        size = parsed['size']
        family = parsed['family']

        if task_type != 'all' and task != task_type:
            continue
        if model_name in ['', 'fusion', 'classification', 'regression']:
            continue

        print(f"\n{'─'*70}")
        print(f"Processing: {folder_name}")
        print(f"  Task: {task} | Model: {model_name} | "
              f"Size: {size} | Family: {family}")

        leakage_dir = folder_path / 'fusion_results' / 'leakage_safe_5fold'
        meta_fusion_dir = folder_path / 'fusion_results' / 'meta_fusion'

        if leakage_dir.exists():
            json_files = list(leakage_dir.glob("*_aggregate_metrics.json"))
            json_files.extend(leakage_dir.glob("*_metrics.json"))
            json_files = sorted(set(json_files))
            print(f"  [leakage_safe_5fold] {len(json_files)} metric JSON(s)")

            for json_file in json_files:
                filename = json_file.name
                method_key = None
                for key, fname in config.fusion_methods.items():
                    if filename == fname:
                        method_key = key
                        break
                if method_key is None:
                    base_name = filename
                    for suffix in ['_aggregate_metrics.json',
                                   '_metrics.json', '.json']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    if base_name.startswith('fuse-'):
                        method_key = base_name
                    else:
                        for key in config.fusion_methods.keys():
                            if key in base_name or base_name in key:
                                method_key = key
                                break
                        if method_key is None:
                            method_key = base_name

                experiments[model_name][method_key] = {
                    'task': task, 'model': model_name, 'size': size,
                    'family': family, 'metrics_file': json_file,
                    'dir': folder_path, 'source': 'leakage_safe_5fold'
                }
                print(f"    ✓ metric: {method_key}  ←  {json_file.name}")
        else:
            print(f"  [leakage_safe_5fold] NOT FOUND")

        if meta_fusion_dir.exists():
            meta_fusion_file = meta_fusion_dir / 'meta_fusion_metrics.json'
            if meta_fusion_file.exists():
                meta_metrics = load_metrics_file(meta_fusion_file)
                if meta_metrics:
                    print(f"  [meta_fusion] reading {meta_fusion_file.name}")
                    print(f"    top-level keys: {list(meta_metrics.keys())}")

                    for strategy in ['average', 'voting', 'stacking',
                                     'weighted', 'confidence_selection', 'best']:
                        if strategy in meta_metrics and \
                                isinstance(meta_metrics[strategy], dict):
                            exp_key = f'ensemble_{strategy}'
                            experiments[model_name][exp_key] = {
                                'task': task, 'model': model_name, 'size': size,
                                'family': family,
                                'metrics_file': meta_fusion_file,
                                'dir': folder_path, 'source': 'meta_fusion',
                                'ensemble_method': strategy
                            }
                            print(f"    ✓ ensemble metric: {strategy}  "
                                  f"(key='{exp_key}')")

                    experiments[model_name]['meta_fusion'] = {
                        'task': task, 'model': model_name, 'size': size,
                        'family': family, 'metrics_file': meta_fusion_file,
                        'dir': folder_path, 'source': 'meta_fusion'
                    }
                    print(f"    ✓ meta_fusion summary row registered")
            else:
                print(f"  [meta_fusion] meta_fusion_metrics.json NOT FOUND")

        print(f"  Scanning prediction CSVs:")
        n1, n2 = discover_predictions_for_model(
            folder_path, model_name, task, size, family, experiments)
        print(f"  → Phase 1: {n1} file(s) from leakage_safe_5fold")
        print(f"  → Phase 2: {n2} file(s) from meta_fusion (fallback)")

    print(f"\n{'='*70}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*70}")

    method_counts = defaultdict(int)
    ensemble_metric_counts = defaultdict(int)
    for model_data in experiments.values():
        for k in model_data.keys():
            if k in ('meta_fusion', 'meta_fusion_summary'):
                continue
            if k.startswith('predictions_'):
                continue
            if k.startswith('ensemble_'):
                ensemble_metric_counts[k] += 1
            else:
                method_counts[k] += 1

    print(f"\nBase/advanced fusion metric JSONs:")
    for category, methods in config.fusion_method_groups.items():
        found = [m for m in methods if m in method_counts]
        if found:
            print(f"  {category}: {len(found)}/{len(methods)}")
            for m in found:
                print(f"    ✓ {get_method_display_name(m, config)}: "
                      f"{method_counts[m]} models")

    if ensemble_metric_counts:
        print(f"\nEnsemble metric JSONs:")
        for k, v in sorted(ensemble_metric_counts.items()):
            print(f"    ✓ {get_method_display_name(k, config)}: {v} models")

    meta_count = sum(1 for md in experiments.values() if 'meta_fusion' in md)
    if meta_count > 0:
        print(f"\n  Meta-Fusion summary: found in {meta_count} models")

    print(f"\nPrediction file coverage per model:")
    for mn, md in sorted(experiments.items()):
        pred_keys = sorted(k.replace('predictions_', '')
                           for k in md.keys() if k.startswith('predictions_'))
        ensemble_preds = [k for k in pred_keys if k.startswith('ensemble_')]
        other_preds = [k for k in pred_keys if not k.startswith('ensemble_')]
        print(f"  {mn}:")
        print(f"    base fusion preds ({len(other_preds)}): {other_preds}")
        print(f"    ensemble preds   ({len(ensemble_preds)}): {ensemble_preds}")

    return dict(experiments)


# =======================================================================
#  DATA AGGREGATION  (JSON + prediction-derived, merged)
# =======================================================================

def aggregate_experiment_results(
    experiments: Dict,
    config: ExperimentConfig,
    dys_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the master dataframe.

    One row per (model, method). Every (model, method) that appears in
    EITHER a metrics JSON OR a prediction CSV becomes a row. Metrics are
    merged in this order:

      1. Compute every metric we can from the prediction CSV (all
         classification metrics + subgroup split by Dys/Typ; regression
         rmse/r2 + subgroup).
      2. Overlay JSON-provided metrics. JSON wins when it provides a
         non-null value (JSON was computed with the study's official
         thresholding and subgroup bookkeeping).
      3. For classification, if JSON only provided combined metrics
         (sensitivity / specificity / ...) with no subgroup, we keep the
         prediction-derived subgroup columns as a fallback — they get
         computed in step 1 and JSON overlay leaves them intact.

    This guarantees that ensemble_* prediction files (ensemble_weighted,
    ensemble_voting, ...) always contribute a row even if
    meta_fusion_metrics.json is missing or lacks that key.
    """
    rows: List[Dict[str, Any]] = []

    for model_name, model_data in experiments.items():
        # --- infer task / size / family from any child entry ---
        task = 'unknown'
        size = 'Unknown'
        family = 'Other'
        for k, v in model_data.items():
            if isinstance(v, dict) and 'task' in v:
                task = v.get('task', 'unknown')
                size = v.get('size', 'Unknown')
                family = v.get('family', 'Other')
                break
        if task == 'unknown':
            continue

        # -----------------------------------------------------------
        # (A) JSON-derived metrics, per method key
        # -----------------------------------------------------------
        json_metrics: Dict[str, Dict[str, float]] = {}
        json_source: Dict[str, str] = {}

        for method_key, method_data in model_data.items():
            if method_key.startswith('predictions_'):
                continue
            metrics_file = method_data.get('metrics_file')
            if metrics_file is None:
                continue

            if method_key.startswith('ensemble_'):
                metrics = load_metrics_file(metrics_file)
                if metrics is None:
                    continue
                ensemble_method = method_data.get('ensemble_method')
                if ensemble_method and ensemble_method in metrics:
                    json_metrics[method_key] = extract_metrics_from_result(
                        metrics[ensemble_method], task)
                    json_source[method_key] = 'meta_fusion'
            else:
                metrics = load_metrics_file(metrics_file)
                if metrics is None:
                    continue
                json_metrics[method_key] = extract_metrics_from_result(
                    metrics, task)
                json_source[method_key] = method_data.get(
                    'source', 'leakage_safe_5fold')

        # -----------------------------------------------------------
        # (B) Prediction-derived metrics, per method key
        # -----------------------------------------------------------
        pred_metrics: Dict[str, Dict[str, float]] = {}
        pred_source: Dict[str, str] = {}

        for method_key, method_data in model_data.items():
            if not method_key.startswith('predictions_'):
                continue
            canonical = method_key.replace('predictions_', '', 1)
            pred_file = method_data.get('predictions_file')
            if pred_file is None or not Path(pred_file).exists():
                continue
            pred_df = load_predictions_file(Path(pred_file))
            if pred_df is None or pred_df.empty:
                continue

            # Inject speaker IDs if the ensemble file lacks them.
            pred_df = _inject_speaker_ids(
                pred_df, method_data.get('dir'))

            computed = compute_metrics_from_predictions(
                pred_df, task, dys_ids)

            if computed:
                pred_metrics[canonical] = computed
                pred_source[canonical] = method_data.get(
                    'source', 'prediction_csv')

                # health check
                has_ids = _detect_prediction_columns(pred_df)[2] is not None
                has_sub = any(k.startswith('subgroup_') for k in computed)
                tag = ('✓ IDs+subgroup' if (has_ids and has_sub)
                       else '✓ IDs, no subgroup' if has_ids
                       else '⚠ NO IDs — subgroup skipped')
                print(f"    [PRED] {model_name} / {canonical}: "
                      f"{len(computed)} metric(s)  {tag}  "
                      f"← {Path(pred_file).name}")

        # -----------------------------------------------------------
        # (C) Union of every method key seen in JSON or predictions
        # -----------------------------------------------------------
        all_methods = set(json_metrics.keys()) | set(pred_metrics.keys())

        for method_key in sorted(all_methods):
            jm = json_metrics.get(method_key, {})
            pm = pred_metrics.get(method_key, {})

            # Start from prediction-derived values, overlay JSON
            merged: Dict[str, float] = dict(pm)
            for k, v in jm.items():
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                merged[k] = v

            # Display label
            if method_key.startswith('ensemble_'):
                ens = method_key.replace('ensemble_', '')
                method_display = config.meta_fusion_display_names.get(
                    ens, ens.replace('_', ' ').title())
            else:
                method_display = get_method_display_name(method_key, config)

            # Source tag
            if method_key in json_metrics and method_key in pred_metrics:
                source = 'json+pred'
            elif method_key in json_metrics:
                source = json_source.get(method_key, 'json')
            else:
                source = pred_source.get(method_key, 'prediction_csv')

            row: Dict[str, Any] = {
                'Task': task, 'Model': model_name, 'Size': size,
                'Family': family, 'Method': method_key,
                'Method_Label': method_display, 'Source': source,
            }

            if task == 'classification':
                all_metrics = (config.classification_metrics
                               + config.classification_subgroup_metrics)
            else:
                all_metrics = (config.regression_metrics
                               + config.regression_subgroup_metrics)

            for metric in all_metrics:
                row[metric] = merged.get(metric, np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = (config.classification_metrics
                    + config.classification_subgroup_metrics
                    + config.regression_metrics
                    + config.regression_subgroup_metrics)
    for metric in numeric_cols:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    return df


# -----------------------------------------------------------------------
#  DIAGNOSTIC: PREDICTION-SCALE (regression)
# -----------------------------------------------------------------------

def diagnose_diverged_predictions(experiments: Dict, output_dir: Path,
                                  config: ExperimentConfig):
    print(f"\n{'='*130}")
    print("PREDICTION-SCALE DIAGNOSTIC (regression)")
    print(f"{'='*130}")

    print(f"{'Model':<14} {'Method':<28} {'n':>5} "
          f"{'obs_min':>9} {'obs_max':>9} "
          f"{'prd_min':>9} {'prd_max':>9} {'prd_mean':>9} "
          f"{'RMSE':>9} {'R2':>9}")
    print("-" * 130)

    for model_name, model_data in experiments.items():
        sample_task = None
        for k, v in model_data.items():
            if isinstance(v, dict) and 'task' in v:
                sample_task = v['task']
                break
        if sample_task != 'regression':
            continue

        for method_key, method_data in model_data.items():
            if not method_key.startswith('predictions_'):
                continue
            pred_file = method_data.get('predictions_file')
            if pred_file is None or not Path(pred_file).exists():
                continue

            df = load_predictions_file(Path(pred_file))
            if df is None or df.empty:
                continue

            obs_col, pred_col, id_col, prob_col = _detect_prediction_columns(df)
            method_short = method_key.replace('predictions_', '')

            if obs_col is None or pred_col is None:
                continue

            o = pd.to_numeric(df[obs_col], errors='coerce').dropna().values
            p = pd.to_numeric(df[pred_col], errors='coerce').dropna().values
            n = min(len(o), len(p))
            if n < 3:
                continue
            o, p = o[:n], p[:n]

            rmse = float(np.sqrt(np.mean((o - p) ** 2)))
            ss_res = np.sum((o - p) ** 2)
            ss_tot = np.sum((o - o.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            print(f"{get_ultra_short_model_name(model_name, config):<14} "
                  f"{method_short:<28} {n:>5} "
                  f"{o.min():>9.3f} {o.max():>9.3f} "
                  f"{p.min():>9.3f} {p.max():>9.3f} {p.mean():>9.3f} "
                  f"{rmse:>9.3f} {r2:>9.3f}")

    print("=" * 130 + "\n")


# =======================================================================
#  SCATTER PLOT (DYS + NORM) — REGRESSION
# =======================================================================

def plot_dys_scatter_audio_text_fusion_single(
    df: pd.DataFrame, experiments: Dict, output_dir: Path,
    config: ExperimentConfig, task_type: str = 'regression',
    dys_ids: List[str] = None,
):
    if task_type != 'regression':
        print("Scatter plots only available for regression tasks")
        return

    plot_df = df[df['Task'] == 'regression'].copy()
    if plot_df.empty:
        print("No regression data found")
        return

    if 'subgroup_rmse' not in plot_df.columns:
        print("⚠ 'subgroup_rmse' column not present — cannot rank candidates.")
        return

    if dys_ids:
        print(f"\nUsing Dys/Norm split with {len(dys_ids)} Dys speaker IDs")
    else:
        print("\n⚠ No Dys speaker IDs provided. Norm subgroup will be empty.")

    models = plot_df['Model'].unique()
    print(f"\nFound {len(models)} models for regression")

    def _validated_subgroup_rmse(model_name, method_key):
        sub = plot_df[(plot_df['Model'] == model_name)
                      & (plot_df['Method'] == method_key)]
        if sub.empty or 'subgroup_rmse' not in sub.columns:
            return None, 0
        vals = pd.to_numeric(sub['subgroup_rmse'], errors='coerce').dropna().values
        if len(vals) == 0:
            return None, 0
        valid = vals[np.isfinite(vals) & (vals > 1e-6)]
        if len(valid) == 0:
            return None, 0
        return float(np.mean(valid)), int(len(valid))

    best_text_model = best_text_method = None
    best_text_rmse = float('inf')
    best_fusion_model = best_fusion_method = None
    best_fusion_rmse = float('inf')

    for model in models:
        for method in plot_df[plot_df['Model'] == model]['Method'].unique():
            rmse_val, _ = _validated_subgroup_rmse(model, method)
            if rmse_val is None:
                continue

            if is_text_embedding_only(method):
                if rmse_val < best_text_rmse:
                    best_text_rmse = rmse_val
                    best_text_model = model
                    best_text_method = method

            if (not is_clinical_feature_only(method)
                    and not is_text_embedding_only(method)
                    and not is_ensemble_method(method)):
                if rmse_val < best_fusion_rmse:
                    best_fusion_rmse = rmse_val
                    best_fusion_model = model
                    best_fusion_method = method

    clinical_method = 'audio_only'
    clinical_model = None
    for model in models:
        strict_matches = [m for m in plot_df[plot_df['Model'] == model]['Method'].unique()
                          if is_clinical_feature_only(m)]
        if strict_matches:
            clinical_model = model
            clinical_method = strict_matches[0]
            break

    print(f"\n{'='*100}")
    print("SCATTER PLOT — METHOD SELECTION (Regression, validated Dys RMSE)")
    print(f"{'='*100}")
    print(f"  Clinical-Feature-Only panel: model={clinical_model}, "
          f"method={clinical_method}")
    if best_text_method:
        print(f"  Best Text-Embedding   panel: model={best_text_model}, "
              f"method={best_text_method} (Dys RMSE={best_text_rmse:.4f})")
    if best_fusion_method:
        print(f"  Best Fusion           panel: model={best_fusion_model}, "
              f"method={best_fusion_method} (Dys RMSE={best_fusion_rmse:.4f})")
    print(f"{'='*100}\n")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 18,
        'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    })

    method_types = [
        ('clinical', 'Clinical-Feature-Only', axes[0],
         clinical_model, clinical_method),
        ('text', 'Text-Embedding-Only', axes[1],
         best_text_model, best_text_method),
        ('fusion', 'Best Fusion', axes[2],
         best_fusion_model, best_fusion_method),
    ]

    all_predictions = {}

    for method_type, display_name, ax, model, method in method_types:
        if model is None or method is None:
            ax.text(0.5, 0.5, f'No model found\nfor {display_name}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=18)
            ax.set_xlabel('Observed', fontsize=18)
            ax.set_ylabel('Predicted', fontsize=18)
            ax.set_title(f'{display_name}\nDys vs Norm', fontsize=20,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue

        pred_df = load_predictions_unfiltered(experiments, model, method)
        if pred_df is None:
            ax.text(0.5, 0.5, 'No predictions file',
                    ha='center', va='center', transform=ax.transAxes, fontsize=18)
            ax.set_xlabel('Observed', fontsize=18)
            ax.set_ylabel('Predicted', fontsize=18)
            ax.set_title(f'{display_name}\nDys vs Norm', fontsize=20,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue

        obs_col, pred_col, id_col, prob_col = _detect_prediction_columns(pred_df)

        if obs_col is None or pred_col is None:
            ax.text(0.5, 0.5, 'No obs/pred columns',
                    ha='center', va='center', transform=ax.transAxes, fontsize=18)
            ax.set_xlabel('Observed', fontsize=18)
            ax.set_ylabel('Predicted', fontsize=18)
            ax.set_title(f'{display_name}\nDys vs Norm', fontsize=20,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue

        if dys_ids and id_col is not None:
            ids_str = pred_df[id_col].astype(str)
            dys_id_strs = [str(x) for x in dys_ids]
            dys_mask = ids_str.isin(dys_id_strs)
            dys_df = pred_df[dys_mask].drop_duplicates(subset=[id_col], keep='first')
            norm_df = pred_df[~dys_mask].drop_duplicates(subset=[id_col], keep='first')
        else:
            if id_col is not None:
                dys_df = pred_df.drop_duplicates(subset=[id_col], keep='first')
            else:
                dys_df = pred_df
            norm_df = pred_df.iloc[0:0]

        def _extract(d):
            if d is None or d.empty:
                return np.array([]), np.array([])
            o = pd.to_numeric(d[obs_col], errors='coerce').values.astype(float)
            p = pd.to_numeric(d[pred_col], errors='coerce').values.astype(float)
            m = ~(np.isnan(o) | np.isnan(p))
            return o[m], p[m]

        dys_o, dys_p = _extract(dys_df)
        norm_o, norm_p = _extract(norm_df)

        if len(dys_o) == 0 and len(norm_o) == 0:
            ax.text(0.5, 0.5, 'No valid samples',
                    ha='center', va='center', transform=ax.transAxes, fontsize=18)
            ax.set_xlabel('Observed', fontsize=18)
            ax.set_ylabel('Predicted', fontsize=18)
            ax.set_title(f'{display_name}\nDys vs Norm', fontsize=20,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue

        def _metrics(o, p):
            if len(o) < 2:
                return (np.nan, np.nan, len(o))
            rmse = float(np.sqrt(np.mean((o - p) ** 2)))
            r2 = float(pearsonr(o, p)[0] ** 2) if len(o) > 2 else np.nan
            return rmse, r2, len(o)

        dys_rmse_plot, dys_r2_plot, n_dys = _metrics(dys_o, dys_p)
        norm_rmse_plot, norm_r2_plot, n_norm = _metrics(norm_o, norm_p)

        if n_dys > 0:
            ax.scatter(dys_o, dys_p, alpha=0.75, s=170, color='#E74C3C',
                       marker='o', edgecolors='black', linewidths=1.4,
                       label=f'Dys  (n={n_dys}, RMSE={dys_rmse_plot:.3f}, '
                             f'R²={dys_r2_plot:.3f})')
        if n_norm > 0:
            ax.scatter(norm_o, norm_p, alpha=0.60, s=150, color='#3498DB',
                       marker='^', edgecolors='black', linewidths=1.2,
                       label=f'Norm (n={n_norm}, RMSE={norm_rmse_plot:.3f}, '
                             f'R²={norm_r2_plot:.3f})')

        all_obs = (np.concatenate([dys_o, norm_o])
                   if (len(dys_o) + len(norm_o)) > 0 else np.array([0.0, 1.0]))
        all_pred = (np.concatenate([dys_p, norm_p])
                    if (len(dys_o) + len(norm_o)) > 0 else np.array([0.0, 1.0]))

        dmin = float(np.nanmin([all_obs.min(), all_pred.min()]))
        dmax = float(np.nanmax([all_obs.max(), all_pred.max()]))
        rng = dmax - dmin if dmax > dmin else 1.0
        pad = 0.05 * rng
        lo, hi = dmin - pad, dmax + pad

        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, linewidth=3,
                label='y = x (perfect)')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Observed', fontsize=18)
        ax.set_ylabel('Predicted', fontsize=18)
        ax.set_title(f'{display_name}\nDys vs Norm', fontsize=20,
                     fontweight='bold')
        ax.legend(loc='upper left', fontsize=13, framealpha=0.95,
                  handletextpad=0.5, borderpad=0.6, labelspacing=0.6)
        ax.grid(True, alpha=0.3)

        all_predictions[method_type] = {
            'model': model, 'method': method,
            'dys': {'obs': dys_o, 'pred': dys_p, 'rmse': dys_rmse_plot,
                    'r2': dys_r2_plot, 'n': n_dys},
            'norm': {'obs': norm_o, 'pred': norm_p, 'rmse': norm_rmse_plot,
                     'r2': norm_r2_plot, 'n': n_norm},
        }

    plt.tight_layout()
    out_png = output_dir / 'dys_scatter_audio_text_fusion_single.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Dys/Norm scatter plot saved to: {out_png}")

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    })

    if all_predictions:
        rows = []
        panel_display = {'clinical': 'Clinical-Feature-Only',
                         'text': 'Text-Embedding-Only',
                         'fusion': 'Best Fusion'}
        for method_type, d in all_predictions.items():
            panel_name = panel_display.get(method_type, method_type)
            for grp_key, grp_label in [('dys', 'Dys'), ('norm', 'Norm')]:
                g = d[grp_key]
                rows.append({
                    'Panel': panel_name, 'Subgroup': grp_label,
                    'Model': get_ultra_short_model_name(d['model'], config),
                    'Model_Full': d['model'],
                    'Method_Used': get_method_display_name(d['method'], config),
                    'Method_Key': d['method'],
                    'Is_Ensemble': is_ensemble_method(d['method']),
                    'RMSE': g['rmse'], 'R²': g['r2'], 'N': g['n'],
                })
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / 'dys_scatter_summary.csv', index=False)
        print(f"✓ Dys/Norm scatter summary saved to: "
              f"{output_dir / 'dys_scatter_summary.csv'}")

    return all_predictions


# =======================================================================
#  TOP-K SELECTION
# =======================================================================

def select_top_k_models(df: pd.DataFrame, k: int, task_type: str = None,
                        config: ExperimentConfig = None) -> pd.DataFrame:
    if config is None:
        config = ExperimentConfig()
    if k <= 0:
        return df

    tasks = [task_type] if task_type else df['Task'].unique()
    selected_models = []

    for task in tasks:
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue

        if task == 'classification':
            ranking_metric = config.ranking_metric_classification
            lower_is_better = False
        else:
            ranking_metric = config.ranking_metric_regression
            lower_is_better = True

        if ranking_metric not in task_df.columns:
            print(f"  Warning: Ranking metric {ranking_metric} not found for {task}")
            continue

        model_performance = task_df.groupby('Model')[ranking_metric].mean().reset_index()
        model_performance = model_performance.dropna()
        if model_performance.empty:
            continue

        model_performance = model_performance.sort_values(
            ranking_metric, ascending=lower_is_better)

        top_models = model_performance.head(k)['Model'].tolist()
        selected_models.extend(top_models)

        print(f"\n  Top {k} models for {task} "
              f"(by {config.metric_labels.get(ranking_metric, ranking_metric)}, "
              f"{'lower' if lower_is_better else 'higher'} is better):")
        for _, row in model_performance.head(k).iterrows():
            print(f"    {row['Model']}: {row[ranking_metric]:.4f}")

    if selected_models:
        return df[df['Model'].isin(selected_models)]
    return df


# =======================================================================
#  DIAGNOSTICS
# =======================================================================

def print_method_coverage(df: pd.DataFrame):
    """Print rows per (Task, Method_Label). Instantly reveals missing methods."""
    if df.empty:
        print("  (empty df — nothing to report)")
        return
    print(f"\n{'='*70}")
    print("METHOD COVERAGE (rows per Task × Method_Label)")
    print(f"{'='*70}")
    pivot = df.groupby(['Task', 'Method_Label']).size().unstack(fill_value=0)
    print(pivot.to_string())
    print(f"{'='*70}\n")


def print_regression_sanity_check(df: pd.DataFrame, config: ExperimentConfig):
    reg = df[df['Task'] == 'regression'].copy()
    if reg.empty:
        print("  No regression rows — sanity check skipped.")
        return

    print(f"\n{'='*70}")
    print("SANITY CHECK — regression metric distributions")
    print(f"{'='*70}")
    print(f"  Total regression rows: {len(reg)}")

    for col in ['rmse', 'r2', 'subgroup_rmse', 'subgroup_r2',
                'non_subgroup_rmse', 'non_subgroup_r2']:
        if col not in reg.columns:
            print(f"\n  {col}: COLUMN MISSING")
            continue
        s = pd.to_numeric(reg[col], errors='coerce')
        n_valid = int(s.notna().sum())
        if n_valid == 0:
            print(f"\n  {col}: all NaN")
            continue
        print(f"\n  {col}:")
        print(f"    n_valid     : {n_valid} / {len(s)}")
        print(f"    min / median / max : {s.min():.4f} / "
              f"{s.median():.4f} / {s.max():.4f}")

    reg['is_ens'] = reg['Method'].apply(is_ensemble_method)
    print(f"\n  Ensemble coverage (regression):")
    print(f"    Ensemble rows          : {reg['is_ens'].sum()}")
    if 'rmse' in reg.columns:
        print(f"    Ensemble rows with rmse: "
              f"{reg.loc[reg['is_ens'], 'rmse'].notna().sum()}")
    if 'r2' in reg.columns:
        print(f"    Ensemble rows with r2  : "
              f"{reg.loc[reg['is_ens'], 'r2'].notna().sum()}")

    print(f"\n  Per-method 'rmse' (regression, ALL rows):")
    print(f"  {'Method_Label':<32} {'n':>4} {'min':>10} {'max':>10} {'mean':>10}")
    print(f"  {'-'*32} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
    for m, g in reg.groupby('Method_Label'):
        vals = pd.to_numeric(g['rmse'], errors='coerce').dropna()
        if len(vals) == 0:
            print(f"  {str(m):<32} {0:>4} {'—':>10} {'—':>10} {'—':>10}")
        else:
            print(f"  {str(m):<32} {len(vals):>4} {vals.min():>10.4f} "
                  f"{vals.max():>10.4f} {vals.mean():>10.4f}")
    print(f"{'='*70}\n")


def print_classification_sanity_check(df: pd.DataFrame, config: ExperimentConfig):
    cls = df[df['Task'] == 'classification'].copy()
    if cls.empty:
        print("  No classification rows — sanity check skipped.")
        return

    print(f"\n{'='*70}")
    print("SANITY CHECK — classification subgroup coverage")
    print(f"{'='*70}")
    print(f"  Total classification rows: {len(cls)}")

    cols = ['sensitivity', 'specificity', 'macro_f1', 'balanced_accuracy',
            'roc_auc',
            'subgroup_sensitivity', 'subgroup_specificity', 'subgroup_macro_f1',
            'subgroup_balanced_accuracy', 'subgroup_roc_auc',
            'non_subgroup_sensitivity', 'non_subgroup_specificity',
            'non_subgroup_macro_f1', 'non_subgroup_balanced_accuracy',
            'non_subgroup_roc_auc']
    for c in cols:
        if c not in cls.columns:
            print(f"  {c}: COLUMN MISSING")
            continue
        n_valid = int(cls[c].notna().sum())
        print(f"  {c:<32}: {n_valid} / {len(cls)} populated")

    print(f"\n  Per-method subgroup coverage:")
    print(f"  {'Method_Label':<32} {'n':>4} "
          f"{'sub_f1':>10} {'sub_sens':>10} {'sub_spec':>10} "
          f"{'typ_f1':>10} {'typ_sens':>10} {'typ_spec':>10}")
    print(f"  {'-'*32} {'-'*4} {'-'*10} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10} {'-'*10}")
    for m, g in cls.groupby('Method_Label'):
        n = len(g)
        sf1 = g['subgroup_macro_f1'].notna().sum() if \
            'subgroup_macro_f1' in g.columns else 0
        ss = g['subgroup_sensitivity'].notna().sum() if \
            'subgroup_sensitivity' in g.columns else 0
        sp = g['subgroup_specificity'].notna().sum() if \
            'subgroup_specificity' in g.columns else 0
        tf1 = g['non_subgroup_macro_f1'].notna().sum() if \
            'non_subgroup_macro_f1' in g.columns else 0
        ts = g['non_subgroup_sensitivity'].notna().sum() if \
            'non_subgroup_sensitivity' in g.columns else 0
        tp = g['non_subgroup_specificity'].notna().sum() if \
            'non_subgroup_specificity' in g.columns else 0
        print(f"  {str(m):<32} {n:>4} {sf1:>10} {ss:>10} {sp:>10} "
              f"{tf1:>10} {ts:>10} {tp:>10}")
    print(f"{'='*70}\n")


# =======================================================================
#  BOOTSTRAP CI
# =======================================================================

def bootstrap_ci(data: np.ndarray, n_iterations: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    if len(data) == 0 or np.all(np.isnan(data)):
        return np.nan, np.nan, np.nan
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan

    n = len(data)
    means = [np.mean(data[np.random.choice(n, n, replace=True)])
             for _ in range(n_iterations)]
    means = np.array(means)

    if np.all(means == means[0]):
        m = means[0]
        return m, m, m
    m = np.mean(means)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    if not np.isfinite(lo):
        lo = m - 0.01
    if not np.isfinite(hi):
        hi = m + 0.01
    return m, lo, hi


# =======================================================================
#  ABLATION ANALYSIS
# =======================================================================

def perform_ablation_analysis_all_patients(df, config, task_type='classification',
                                           verbose=True):
    if task_type == 'classification':
        metric_name = config.ranking_metric_classification
        lower_is_better = False
    else:
        metric_name = config.ranking_metric_regression
        lower_is_better = True

    task_df = df[df['Task'] == task_type].copy()
    if task_df.empty:
        return {}

    is_ensemble = task_df.apply(
        lambda row: is_ensemble_method(row.get('Method'))
                    or is_ensemble_method(row.get('Method_Label')),
        axis=1,
    )

    if verbose:
        ens_in = task_df[is_ensemble]['Method_Label'].unique().tolist()
        print(f"  Ensemble/meta-fusion rows present (kept for ranking): "
              f"{sorted(ens_in)}")

    if metric_name not in task_df.columns:
        if verbose:
            print(f"  Warning: metric '{metric_name}' not found for {task_type}")
        return {}

    work = task_df.copy()
    work[metric_name] = pd.to_numeric(work[metric_name], errors='coerce')
    n_before = len(work)
    work = work[work[metric_name].notna()].copy()

    if task_type == 'regression':
        n_before_plausible = len(work)
        work = work[
            work[metric_name].apply(
                lambda v: is_plausible_regression_metric(v, metric_name)
            )
        ].copy()
        if verbose:
            print(f"\n  [ABLATION-VALIDATE] metric='{metric_name}' "
                  f"({'lower' if lower_is_better else 'higher'} is better)")
            print(f"    Rows before filter              : {n_before}")
            print(f"    Rows after NaN filter           : {n_before_plausible}")
            print(f"    Rows after plausibility filter  : {len(work)}")

    if work.empty:
        if verbose:
            print(f"  Warning: No rows with plausible {metric_name} "
                  f"for {task_type}")
        return {}

    base_df = work
    all_methods = base_df['Method_Label'].unique()

    if verbose:
        print(f"\n  Base methods with plausible {metric_name} for {task_type}: "
              f"{len(all_methods)}")
        for m in sorted(all_methods):
            print(f"    ✓ {m}")

    baseline_mask = base_df['Method_Label'].apply(is_baseline_method)
    ranking_pool = base_df[~baseline_mask]

    if ranking_pool.empty:
        if verbose:
            print("  ⚠ No non-baseline methods left for reference ranking — "
                  "falling back to full pool")
        ranking_pool = base_df

    method_performance = ranking_pool.groupby('Method_Label')[metric_name].mean()

    if verbose:
        print(f"\n  [ABLATION-RANK] Method mean {metric_name} "
              f"({'lower' if lower_is_better else 'higher'} is better) "
              f"[baselines excluded from pool]:")
        ranked = (method_performance.sort_values(ascending=True)
                  if lower_is_better
                  else method_performance.sort_values(ascending=False))
        for m, v in ranked.items():
            print(f"    {v:+.4f}  {m}")

        excluded_baselines = base_df[baseline_mask]['Method_Label'].unique().tolist()
        if excluded_baselines:
            print(f"\n  [ABLATION-RANK] Baselines excluded from reference pool:")
            for b in excluded_baselines:
                bval = base_df[base_df['Method_Label'] == b][metric_name].mean()
                print(f"    {bval:+.4f}  {b}  (baseline)")

    if lower_is_better:
        best_method = method_performance.idxmin()
        best_score = method_performance.min()
    else:
        best_method = method_performance.idxmax()
        best_score = method_performance.max()

    best_cell_rows = base_df[base_df['Method_Label'] == best_method]
    best_cell = (best_cell_rows[metric_name].min() if lower_is_better
                 else best_cell_rows[metric_name].max())

    if verbose:
        print(f"\n  Best base method (by mean): {best_method}")
        print(f"    mean across models : {best_score:+.4f}")
        print(f"    best single cell   : {best_cell:+.4f}")
        print(f"    metric             : {metric_name}, "
              f"{'lower' if lower_is_better else 'higher'} is better")

    ablation_results = []
    skipped_methods = []

    for method in all_methods:
        if method == best_method:
            continue

        method_df = base_df[base_df['Method_Label'] == method]
        best_df = base_df[base_df['Method_Label'] == best_method]

        method_values, best_values, paired_models = [], [], []
        for model in method_df['Model'].unique():
            m_val = method_df[method_df['Model'] == model][metric_name].values
            b_val = best_df[best_df['Model'] == model][metric_name].values
            if len(m_val) > 0 and len(b_val) > 0 \
                    and not np.isnan(m_val[0]) and not np.isnan(b_val[0]):
                method_values.append(m_val[0])
                best_values.append(b_val[0])
                paired_models.append(model)

        if len(method_values) < 2:
            skipped_methods.append((method, f"only {len(method_values)} paired samples"))
            continue

        method_values = np.array(method_values)
        best_values = np.array(best_values)

        if lower_is_better:
            diff = method_values - best_values
        else:
            diff = best_values - method_values

        try:
            t_stat, p_value_ttest = ttest_rel(best_values, method_values)
        except Exception:
            t_stat, p_value_ttest = np.nan, np.nan

        try:
            _, p_value_wilcoxon = wilcoxon(best_values, method_values)
        except Exception:
            p_value_wilcoxon = np.nan

        pooled_std = np.sqrt(
            (np.std(best_values, ddof=1) ** 2
             + np.std(method_values, ddof=1) ** 2) / 2
        )
        effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else np.nan

        ablation_results.append({
            'Removed_Component': method,
            'Best_Method': best_method,
            'Best_Score': np.mean(best_values),
            'Removed_Score': np.mean(method_values),
            'Difference': np.mean(diff),
            'Difference_Std': np.std(diff),
            't_statistic': t_stat,
            'p_value_ttest': p_value_ttest,
            'p_value_wilcoxon': p_value_wilcoxon,
            'effect_size': effect_size,
            'n_pairs': len(method_values),
            'paired_models': paired_models,
            'significant': (p_value_ttest < 0.05
                            if not np.isnan(p_value_ttest) else False),
        })

    if verbose and skipped_methods:
        print(f"\n  Skipped methods (insufficient paired data):")
        for method, reason in skipped_methods:
            print(f"    ✗ {method}: {reason}")

    ablation_df = pd.DataFrame(ablation_results)
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values('Difference', ascending=False)

    return {
        'best_method': best_method,
        'best_score': best_score,
        'best_cell': best_cell,
        'ablation_results': ablation_df,
        'metric': metric_name,
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'skipped_methods': skipped_methods,
    }


# =======================================================================
#  ROBUSTNESS SUMMARY
# =======================================================================

def plot_robustness_summary_all_patients(df: pd.DataFrame, output_dir: Path,
                                         config: ExperimentConfig,
                                         task_type: str = 'classification',
                                         n_iterations: int = 1000):
    if df.empty:
        return None, None

    task_df = df[df['Task'] == task_type]
    if task_df.empty:
        return None, None

    ci_df = None
    ablation_data = None

    if task_type == 'classification':
        metric = config.ranking_metric_classification
        lower_is_better = False
    else:
        metric = config.ranking_metric_regression
        lower_is_better = True

    try:
        print(f"\n[CI] Computing cross-model CIs for {task_type} "
              f"(metric={metric})...")
        ci_rows = []
        for method_label, g in task_df.groupby('Method_Label'):
            vals = pd.to_numeric(g[metric], errors='coerce').dropna().values
            if task_type == 'regression':
                vals = np.array([v for v in vals
                                 if is_plausible_regression_metric(v, metric)])
            if len(vals) < 2:
                continue
            mean, lower, upper = bootstrap_ci(vals, n_iterations=n_iterations,
                                              ci=0.95)
            ci_rows.append({
                'Method_Label': method_label,
                f'{metric}_mean': mean,
                f'{metric}_lower_ci': lower,
                f'{metric}_upper_ci': upper,
                f'{metric}_std': np.nanstd(vals, ddof=1),
                'n_models': len(vals),
                'models': list(g['Model'].unique()),
            })
        ci_df = pd.DataFrame(ci_rows)

        if not ci_df.empty:
            ci_df.to_csv(output_dir / f'confidence_intervals_{task_type}.csv',
                         index=False)
            print(f"✓ Confidence intervals data saved")
    except Exception as e:
        print(f"⚠ Error computing confidence intervals: {e}")

    try:
        print(f"\n[ABLATION] Running ablation for {task_type}...")
        ablation_data = perform_ablation_analysis_all_patients(
            task_df, config, task_type, verbose=True,
        )
        if ablation_data and not ablation_data.get('ablation_results',
                                                    pd.DataFrame()).empty:
            plot_ablation_results_all_patients(ablation_data, output_dir,
                                               config, task_type)
    except Exception as e:
        import traceback
        print(f"⚠ Error computing ablation: {e}")
        traceback.print_exc()

    return ci_df, ablation_data


# =======================================================================
#  NAME SHORTENERS
# =======================================================================

def get_short_model_name(model_name: str, config: ExperimentConfig) -> str:
    if model_name in config.model_name_mapping:
        return config.model_name_mapping[model_name]
    for key, value in config.model_name_mapping.items():
        if key in model_name or model_name in key:
            return value
    cleaned = model_name
    for prefix in ['microsoft.', 'google.', 'facebook/', 'allenai/',
                   'emilyalsentzer/']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    for suffix in ['-uncased', '-cased', '-v2', '-base', '-large', '-small']:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    return cleaned.title()


def get_ultra_short_model_name(model_name: str, config: ExperimentConfig) -> str:
    name = get_short_model_name(model_name, config)
    ll = name.lower()
    family_rules = [
        ('bioclinicalbert', 'BioClin'), ('bio_clinicalbert', 'BioClin'),
        ('clinicalbert', 'ClinBERT'), ('pubmedbert', 'PubMed'),
        ('pubmed', 'PubMed'), ('biomed-roberta', 'BioMedR'),
        ('biomednlp-roberta', 'BioMedR'), ('biomed', 'BioMed'),
        ('scibert', 'SciBERT'), ('legalbert', 'LegalB'),
        ('distilroberta', 'DistilR'), ('distilbert', 'Distil'),
        ('deberta', 'DeBERTa'), ('roberta', 'RoBERTa'),
        ('albert', 'ALBERT'), ('bert', 'BERT'),
    ]
    family = None
    for needle, label in family_rules:
        if needle in ll:
            family = label
            break
    size = ''
    if 'large' in ll:
        size = '-L'
    elif 'base' in ll:
        size = '-B'
    elif 'small' in ll:
        size = '-S'
    if family:
        return f"{family}{size}"
    token = re.split(r'[ _\-]', name)[0]
    return token[:10] + size


# =======================================================================
#  ABLATION PLOT (single-column, no overlap)
# =======================================================================

def plot_ablation_results_all_patients(ablation_data: Dict, output_dir: Path,
                                       config: ExperimentConfig,
                                       task_type: str = 'classification'):
    """
    Two stacked panels for a single-column figure.

    Top    : performance drop (Δ metric) vs the reference.
    Bottom : −log10(p) from the paired t-test.

    Method names are drawn on the TOP panel only. The y-axis is NOT
    shared — both panels use identical y positions computed from the same
    array, which avoids the shared-inverted-axis matplotlib trap.
    """
    ablation_df = ablation_data.get('ablation_results')
    if ablation_df is None or ablation_df.empty:
        print("Warning: No ablation data to plot")
        return

    metric_name = ablation_data['metric_name']
    metric_label = config.metric_labels.get(metric_name, metric_name.upper())
    best_method = ablation_data['best_method']
    best_score = ablation_data['best_score']
    best_cell = ablation_data.get('best_cell', best_score)
    lower_is_better = ablation_data['lower_is_better']

    # ---- Select bars ----
    is_ens = ablation_df['Removed_Component'].apply(is_ensemble_method)
    is_ref = (ablation_df['Removed_Component'].str.lower()
              == str(best_method).lower())

    base_only = ablation_df[~is_ens & ~is_ref].copy()
    if base_only.empty:
        plot_df = ablation_df[~is_ref].copy()
    else:
        plot_df = base_only

    if plot_df.empty:
        print("Warning: Nothing left to plot")
        return

    plot_df = plot_df.sort_values('Difference',
                                  ascending=False).reset_index(drop=True)
    plot_df = plot_df.dropna(subset=['Difference',
                                     'p_value_ttest']).reset_index(drop=True)
    if plot_df.empty:
        print("Warning: No valid ablation rows to plot")
        return

    n_bars = len(plot_df)
    # The y-positions are used by BOTH panels — this is what keeps them
    # aligned without relying on shared axes.
    y_pos = np.arange(n_bars)

    # ---- Figure sizing ----
    fig_height = max(3.6, 0.22 * n_bars + 1.6)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.5, fig_height),
        sharex=False, sharey=False,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.28},
    )
    fig.subplots_adjust(
        left=0.52, right=0.96, top=0.93, bottom=0.10,
    )

    # =========================================================
    #  Panel (a) — Performance drop
    # =========================================================
    colors = ['#2ECC71' if sig else '#E74C3C'
              for sig in plot_df['significant']]
    bars = ax1.barh(y_pos, plot_df['Difference'].values, color=colors,
                    alpha=0.88, height=0.62,
                    edgecolor='black', linewidth=0.5)

    xmax1 = max(0.01, plot_df['Difference'].abs().max())
    pad1 = 0.04 * xmax1
    for bar, val in zip(bars, plot_df['Difference'].values):
        if val >= 0:
            ax1.text(bar.get_width() + pad1,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', ha='left', fontsize=6.0)
        else:
            ax1.text(bar.get_width() - pad1,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', ha='right', fontsize=6.0)

    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=0.9)
    ax1.set_xlabel(f'Δ{metric_label} vs reference',
                   fontsize=7.0, labelpad=2)
    ax1.set_title(
        f'(a) Ablation vs reference ({best_method} = {best_score:.3f})',
        fontsize=7.2, fontweight='bold', pad=3,
    )
    ax1.grid(True, alpha=0.30, axis='x', linewidth=0.4)
    ax1.tick_params(axis='x', labelsize=6.0, pad=2, length=2.0)
    ax1.set_ylim(n_bars - 0.5, -0.5)   # inverted y-axis without sharey

    # Method names on panel (a) only
    for y, name in zip(y_pos, plot_df['Removed_Component'].values):
        ax1.text(-0.02, y, name,
                 transform=ax1.get_yaxis_transform(),
                 ha='right', va='center', fontsize=6.0)
    ax1.set_yticks([])

    from matplotlib.patches import Patch
    legend1 = [
        Patch(facecolor='#2ECC71', alpha=0.88, edgecolor='black',
              linewidth=0.4, label='Significant (p < 0.05)'),
        Patch(facecolor='#E74C3C', alpha=0.88, edgecolor='black',
              linewidth=0.4, label='Not Significant'),
    ]
    ax1.legend(handles=legend1, loc='lower right', fontsize=5.6,
               framealpha=0.95, handletextpad=0.35, borderpad=0.30,
               labelspacing=0.20, handlelength=1.2)

    # =========================================================
    #  Panel (b) — Statistical significance
    # =========================================================
    p_values = plot_df['p_value_ttest'].fillna(1.0).values
    log_p = -np.log10(np.clip(p_values, 1e-300, 1.0))
    colors2 = ['#2ECC71' if sig else '#E74C3C'
               for sig in plot_df['significant']]
    bars2 = ax2.barh(y_pos, log_p, color=colors2, alpha=0.88,
                     height=0.62, edgecolor='black', linewidth=0.5)

    xmax2 = max(0.5, log_p.max() * 1.20)
    pad2 = 0.04 * xmax2
    for bar, p_val in zip(bars2, p_values):
        label = f'{p_val:.3f}' if p_val >= 0.001 else f'{p_val:.1e}'
        ax2.text(bar.get_width() + pad2,
                 bar.get_y() + bar.get_height() / 2,
                 label, va='center', ha='left', fontsize=6.0,
                 fontweight='bold')

    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--',
                alpha=0.8, linewidth=1.0, label='p = 0.05')
    ax2.set_xlim(0, xmax2)
    ax2.set_xlabel('−log10(p-value)  (paired t-test vs reference)',
                   fontsize=7.0, labelpad=2)
    ax2.set_title('(b) Statistical significance',
                  fontsize=7.2, fontweight='bold', pad=3)
    ax2.grid(True, alpha=0.30, axis='x', linewidth=0.4)
    ax2.tick_params(axis='x', labelsize=6.0, pad=2, length=2.0)
    ax2.legend(loc='lower right', fontsize=5.6, framealpha=0.95,
               handletextpad=0.35, borderpad=0.30, labelspacing=0.20,
               handlelength=1.2)
    ax2.set_ylim(n_bars - 0.5, -0.5)   # same inversion, panel-local
    ax2.set_yticks([])
    ax2.set_ylabel('')

    # ---- Save ----
    out_png = output_dir / f'ablation_analysis_all_patients_{task_type}.png'
    out_pdf = output_dir / f'ablation_analysis_all_patients_{task_type}.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Ablation figure saved: {out_png}")

    # ---- Sidecar CSV ----
    summary_cols = ['Removed_Component', 'Removed_Score', 'Difference',
                    'p_value_ttest', 'significant', 'effect_size', 'n_pairs']
    summary_cols = [c for c in summary_cols if c in plot_df.columns]
    plot_df[summary_cols].round(4).to_csv(
        output_dir / f'ablation_summary_all_patients_{task_type}.csv',
        index=False)

    with open(output_dir / f'ablation_reference_{task_type}.txt', 'w') as f:
        f.write(f"Reference method: {best_method}\n")
        f.write(f"Reference {metric_label} (mean across models): "
                f"{best_score:+.4f}\n")
        f.write(f"Reference {metric_label} (best single cell): "
                f"{best_cell:+.4f}\n")
        f.write(f"Direction: "
                f"{'lower is better' if lower_is_better else 'higher is better'}\n")
        f.write(f"Base methods plotted ({len(plot_df)}):\n")
        for comp in plot_df['Removed_Component'].tolist():
            f.write(f"  - {comp}\n")


# =======================================================================
#  OTHER VISUALIZATION FUNCTIONS
# =======================================================================

def plot_method_comparison(df, metric, output_dir, config, title=None,
                           subgroup_prefix=None, task_type='classification',
                           hide_ensembles=False):
    if metric not in df.columns or df.empty:
        return
    plot_df = df[df['Task'] == task_type].copy()
    if hide_ensembles:
        plot_df = plot_df[~plot_df['Method'].apply(is_ensemble_method)]
    if plot_df.empty:
        return

    plot_df['Model_Short'] = plot_df['Model'].apply(
        lambda x: get_ultra_short_model_name(x, config))
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    pivot = plot_df.pivot_table(index='Model_Short', columns=method_col,
                                values=metric, aggfunc='mean')
    if pivot.empty:
        return

    lower_is_better = metric in ['rmse']
    best_vals = pivot.min(axis=1) if lower_is_better else pivot.max(axis=1)
    pivot = pivot.loc[best_vals.sort_values(ascending=lower_is_better).index]

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot.index) * 0.4)))
    pivot.plot(kind='barh', ax=ax, width=0.8, colormap='viridis')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)

    metric_label = config.metric_labels.get(metric, metric.upper())
    title_text = title or (f'{metric_label} by Model and Fusion Method '
                           f'({task_type.title()})')
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Model')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    suffix2 = "_noens" if hide_ensembles else ""
    plt.savefig(output_dir /
                f'method_comparison_{metric}{suffix}{suffix2}_{task_type}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Method comparison saved")


def plot_heatmap(df, metric, output_dir, config, subgroup_prefix=None,
                 task_type='classification', hide_ensembles=False):
    if metric not in df.columns or df.empty:
        return
    plot_df = df[df['Task'] == task_type].copy()
    if hide_ensembles:
        plot_df = plot_df[~plot_df['Method'].apply(is_ensemble_method)]
    if plot_df.empty:
        return

    plot_df['Model_Short'] = plot_df['Model'].apply(
        lambda x: get_ultra_short_model_name(x, config))
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    pivot = plot_df.pivot_table(index=method_col, columns='Model_Short',
                                values=metric, aggfunc='mean')
    if pivot.empty:
        return
    pivot = pivot.dropna(axis=1, how='all')
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.6),
                                    max(8, len(pivot.index) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': config.metric_labels.get(metric, metric.upper())},
                linewidths=0.5, linecolor='white', ax=ax, annot_kws={'fontsize': 8})

    metric_label = config.metric_labels.get(metric, metric.upper())
    title_text = f'{metric_label} Comparison Across Models ({task_type.title()})'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Fusion Method')
    plt.tight_layout()
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    suffix2 = "_noens" if hide_ensembles else ""
    plt.savefig(output_dir /
                f'heatmap_{metric}{suffix}{suffix2}_{task_type}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved")


def plot_sens_spec_three_marker(df, output_dir, config,
                                task_type='classification'):
    """
    Three-marker Sens/Spec plot — single-column width, legend below.

    Blocks (Baseline / Fusion / Ensemble) are separated by dashed horizontal
    lines. No textual block labels appear next to them — the caption
    identifies the groups.

    Markers:
        ● Dys         (red circle)
        ◆ Combined    (gray diamond)
        ▲ Typ         (blue triangle)

    A '*' is appended to methods whose subgroup values are fallbacks from
    the combined metrics (i.e. subgroup data was unavailable).
    """
    plot_df = df[df['Task'] == task_type].copy()

    required_cols = [
        'subgroup_sensitivity', 'non_subgroup_sensitivity', 'sensitivity',
        'subgroup_specificity', 'non_subgroup_specificity', 'specificity',
    ]
    missing = [c for c in required_cols if c not in plot_df.columns]
    if plot_df.empty or missing:
        print(f"plot_sens_spec_three_marker: missing columns {missing}")
        return None

    # ---- Aggregate to one row per method ----
    grouped = plot_df.groupby('Method_Label').agg({
        'subgroup_sensitivity':      'mean',
        'non_subgroup_sensitivity':  'mean',
        'sensitivity':               'mean',
        'subgroup_specificity':      'mean',
        'non_subgroup_specificity':  'mean',
        'specificity':               'mean',
    }).reset_index()

    # Drop the aggregate Meta-Fusion row — it is not a real method.
    grouped = grouped[grouped['Method_Label'].str.lower() != 'meta-fusion']

    # Combined metrics are mandatory.
    grouped = grouped.dropna(subset=['sensitivity', 'specificity'], how='any')

    if grouped.empty:
        print("plot_sens_spec_three_marker: no rows with combined sens/spec")
        return None

    # ---- Flag methods missing true subgroup data BEFORE filling ----
    subgroup_cols = ['subgroup_sensitivity', 'non_subgroup_sensitivity',
                     'subgroup_specificity', 'non_subgroup_specificity']
    grouped['_missing_subgroup'] = (
        grouped[subgroup_cols].isna().any(axis=1)
    )

    # Fallback: if a subgroup value is missing, use the combined value.
    fallback_pairs = [
        ('subgroup_sensitivity',     'sensitivity'),
        ('non_subgroup_sensitivity', 'sensitivity'),
        ('subgroup_specificity',     'specificity'),
        ('non_subgroup_specificity', 'specificity'),
    ]
    for sub_col, comb_col in fallback_pairs:
        grouped[sub_col] = grouped[sub_col].fillna(grouped[comb_col])

    # ---- Three-way classification ----
    def classify(label):
        ll = str(label).lower().strip()

        # Baseline
        if ('clinical-feature-only' in ll
                or 'text-embedding-only' in ll
                or 'audio-only' in ll
                or 'text-only' in ll
                or ll == 'clinical'
                or ll == 'text'):
            return 'Baseline'

        # Ensemble
        if ('ensemble' in ll
                or 'meta-fusion' in ll
                or 'metafusion' in ll
                or 'confidence selection' in ll
                or 'best method' in ll):
            return 'Ensemble'

        # Everything else
        return 'Fusion'

    grouped['Block'] = grouped['Method_Label'].apply(classify)

    # ---- Order within each block ----
    block_order = ['Baseline', 'Fusion', 'Ensemble']
    grouped['BlockRank'] = grouped['Block'].apply(
        lambda b: block_order.index(b) if b in block_order else 99)

    grouped = grouped.sort_values(
        ['BlockRank', 'sensitivity'],
        ascending=[True, False],
    ).reset_index(drop=True)

    methods       = grouped['Method_Label'].tolist()
    blocks        = grouped['Block'].tolist()
    missing_flags = grouped['_missing_subgroup'].tolist()
    y = np.arange(len(methods))

    y_labels = [
        f"{m} *" if miss else m
        for m, miss in zip(methods, missing_flags)
    ]

    # ---- Single-column figure ----
    fig, (ax_sen, ax_spec) = plt.subplots(
        1, 2,
        figsize=(3.5, max(3.2, 0.22 * len(methods))),
        sharey=True,
    )

    panels = [
        (ax_sen,  'subgroup_sensitivity', 'non_subgroup_sensitivity',
         'sensitivity', 'Sensitivity'),
        (ax_spec, 'subgroup_specificity', 'non_subgroup_specificity',
         'specificity', 'Specificity'),
    ]

    for ax, dys_col, typ_col, comb_col, title in panels:
        dys_vals      = grouped[dys_col].values
        typ_vals      = grouped[typ_col].values
        combined_vals = grouped[comb_col].values

        # Connecting segment Typ — Combined — Dys
        for yi, d, t, c in zip(y, dys_vals, typ_vals, combined_vals):
            pts = sorted([(t, '#3498DB'), (c, '#7F8C8D'), (d, '#E74C3C')],
                         key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ax.plot(xs, [yi] * 3, color='#95A5A6', linewidth=0.8,
                    alpha=0.75, zorder=1, solid_capstyle='round')

        # Markers
        ax.scatter(dys_vals, y, s=22, marker='o', color='#E74C3C',
                   edgecolor='black', linewidth=0.5, zorder=3, label='Dys')
        ax.scatter(combined_vals, y, s=20, marker='D', color='#7F8C8D',
                   edgecolor='black', linewidth=0.5, zorder=4, label='Combined')
        ax.scatter(typ_vals, y, s=22, marker='^', color='#3498DB',
                   edgecolor='black', linewidth=0.5, zorder=3, label='Typ')

        # Dashed separators between blocks
        for i in range(1, len(y)):
            if blocks[i] != blocks[i - 1]:
                ax.axhline(i - 0.5, color='#34495E', linewidth=0.7,
                           linestyle='--', zorder=0, alpha=0.75)

        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='x', labelsize=6, pad=1.5, length=2.0)
        ax.set_xlabel(title, fontsize=7, fontweight='bold', labelpad=2)
        ax.grid(True, alpha=0.25, axis='x', linewidth=0.4)

    # ---- Y-tick labels on the sensitivity panel only ----
    ax_sen.set_yticks(y)
    ax_sen.set_yticklabels(y_labels, fontsize=6)
    ax_sen.set_ylim(-0.7, len(methods) - 0.3)
    ax_sen.invert_yaxis()
    ax_sen.tick_params(axis='y', pad=2, length=2.0)

    # ---- Shared legend below both panels ----
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#E74C3C', markeredgecolor='black',
               markeredgewidth=0.5, markersize=4, label='Dys'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='#7F8C8D', markeredgecolor='black',
               markeredgewidth=0.5, markersize=4, label='Combined'),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='#3498DB', markeredgecolor='black',
               markeredgewidth=0.5, markersize=4, label='Typ'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False,
               fontsize=6, handletextpad=0.3, columnspacing=1.0)

    # ---- Footnote for asterisked methods ----
    if any(missing_flags):
        fig.text(0.5, -0.08,
                 '* combined sensitivity/specificity shown; '
                 'subgroup data unavailable for this method.',
                 fontsize=5, color='#2C3E50', ha='center', va='top')

    ax_sen.set_title(
        f'Dys, Combined, and Typ Performance across Fusion Methods '
        f'({task_type.title()})',
        fontsize=7, fontweight='bold', pad=4,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'sens_spec_three_marker_{task_type}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'sens_spec_three_marker_{task_type}.pdf',
                bbox_inches='tight')
    plt.close()

    # ---- Log summary ----
    print(f"✓ Three-marker plot saved: "
          f"sens_spec_three_marker_{task_type}.png/.pdf")
    print(f"  Blocks:")
    for block_name in block_order:
        names = grouped[grouped['Block'] == block_name]['Method_Label'].tolist()
        if names:
            print(f"    {block_name} ({len(names)}): {names}")
    if any(missing_flags):
        starred = [m for m, miss in zip(methods, missing_flags) if miss]
        print(f"  Methods with * (subgroup data missing): {starred}")

    grouped.to_csv(
        output_dir / f'sens_spec_three_marker_data_{task_type}.csv',
        index=False)
    return grouped

def plot_sens_spec_compact(df, output_dir, config, task_type='classification'):
    """
    Compact one-row-per-method Sens/Spec plot.

    Per method:
      ● / ■ / ▲  = marker shape encodes metric
                    (circle = Sensitivity, square = Specificity)
      Red        = Dys subgroup
      Gray       = Pooled (combined)
      Blue       = Typ subgroup

    Sensitivity markers sit slightly above the row centre, Specificity
    markers slightly below. Value labels are placed above the Sensitivity
    markers and below the Specificity markers so the two clusters never
    collide.

    Blocks (Baseline / Fusion / Ensemble) are separated by dashed
    horizontal lines. Methods whose subgroup values were unavailable
    (fallback to combined) are drawn with the Pooled marker only and
    flagged with '*' in the y-label.
    """
    from matplotlib.lines import Line2D

    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print("plot_sens_spec_compact: no rows")
        return None

    required = [
        'sensitivity', 'specificity',
        'subgroup_sensitivity', 'subgroup_specificity',
        'non_subgroup_sensitivity', 'non_subgroup_specificity',
    ]
    missing = [c for c in required if c not in plot_df.columns]
    if missing:
        print(f"plot_sens_spec_compact: missing columns {missing}")
        return None

    grouped = plot_df.groupby('Method_Label').agg({
        'sensitivity':                'mean',
        'specificity':                'mean',
        'subgroup_sensitivity':       'mean',
        'subgroup_specificity':       'mean',
        'non_subgroup_sensitivity':   'mean',
        'non_subgroup_specificity':   'mean',
    }).reset_index()

    # Drop the aggregate Meta-Fusion row — not a real method.
    grouped = grouped[grouped['Method_Label'].str.lower() != 'meta-fusion']

    # Combined metrics are mandatory.
    grouped = grouped.dropna(subset=['sensitivity', 'specificity'], how='any')
    if grouped.empty:
        print("plot_sens_spec_compact: no rows with combined sens/spec")
        return None

    # Flag missing subgroup BEFORE falling back.
    sub_cols = ['subgroup_sensitivity', 'subgroup_specificity',
                'non_subgroup_sensitivity', 'non_subgroup_specificity']
    grouped['_missing_subgroup'] = grouped[sub_cols].isna().any(axis=1)

    # Fallback: fill missing subgroup with combined so the marker exists.
    for sub_c, comb_c in [('subgroup_sensitivity',     'sensitivity'),
                          ('subgroup_specificity',     'specificity'),
                          ('non_subgroup_sensitivity', 'sensitivity'),
                          ('non_subgroup_specificity', 'specificity')]:
        grouped[sub_c] = grouped[sub_c].fillna(grouped[comb_c])

    # ---- Block classification ----
    def _classify(label):
        ll = str(label).lower()
        if ('clinical-feature-only' in ll or 'text-embedding-only' in ll
                or 'audio-only' in ll or 'text-only' in ll
                or ll in ('clinical', 'text')):
            return 'Baseline'
        if ('ensemble' in ll or 'meta-fusion' in ll or 'metafusion' in ll
                or 'confidence selection' in ll or 'best method' in ll):
            return 'Ensemble'
        return 'Fusion'

    grouped['Block'] = grouped['Method_Label'].apply(_classify)
    block_order = ['Baseline', 'Fusion', 'Ensemble']
    grouped['BlockRank'] = grouped['Block'].apply(
        lambda b: block_order.index(b) if b in block_order else 99)

    # Order: block, then within-block by combined sensitivity descending.
    grouped = grouped.sort_values(
        ['BlockRank', 'sensitivity'],
        ascending=[True, False],
    ).reset_index(drop=True)

    methods = grouped['Method_Label'].tolist()
    blocks  = grouped['Block'].tolist()
    flags   = grouped['_missing_subgroup'].tolist()
    y       = np.arange(len(methods))

    y_labels = [f"{m} *" if f else m for m, f in zip(methods, flags)]

    # ---- Figure ----
    fig, ax = plt.subplots(
        figsize=(6.0, max(3.0, 0.30 * len(methods))),
    )

    C_DYS  = '#C0392B'   # red
    C_POOL = '#7F8C8D'   # gray
    C_TYP  = '#2980B9'   # blue

    Y_OFFSET = 0.16           # how far above/below the row centre
    LABEL_OFFSET = 0.30       # how far the value labels sit from the centre

    for i, row in grouped.iterrows():
        yy = y[i]

        if flags[i]:
            # No subgroup — draw Pooled markers only.
            ax.scatter([row['sensitivity']], [yy - Y_OFFSET],
                       marker='o', s=42, color=C_POOL,
                       edgecolor='black', linewidth=0.6, zorder=3)
            ax.scatter([row['specificity']], [yy + Y_OFFSET],
                       marker='s', s=38, color=C_POOL,
                       edgecolor='black', linewidth=0.6, zorder=3)

            ax.text(row['sensitivity'], yy - LABEL_OFFSET,
                    f"{row['sensitivity']:.3f}",
                    ha='center', va='bottom', fontsize=6.2, color=C_POOL)
            ax.text(row['specificity'], yy + LABEL_OFFSET,
                    f"{row['specificity']:.3f}",
                    ha='center', va='top', fontsize=6.2, color=C_POOL)
        else:
            # Full Dys / Pooled / Typ cluster.
            # --- Sensitivity (top sub-row, circles) ---
            xs_sens = [row['subgroup_sensitivity'],
                       row['sensitivity'],
                       row['non_subgroup_sensitivity']]
            ax.plot([min(xs_sens), max(xs_sens)], [yy - Y_OFFSET] * 2,
                    color='#D5D8DC', linewidth=0.9, zorder=1)
            ax.scatter(xs_sens, [yy - Y_OFFSET] * 3,
                       marker='o',
                       s=[42, 44, 42],
                       c=[C_DYS, C_POOL, C_TYP],
                       edgecolor='black', linewidth=0.6, zorder=3)

            # --- Specificity (bottom sub-row, squares) ---
            xs_spec = [row['subgroup_specificity'],
                       row['specificity'],
                       row['non_subgroup_specificity']]
            ax.plot([min(xs_spec), max(xs_spec)], [yy + Y_OFFSET] * 2,
                    color='#D5D8DC', linewidth=0.9, zorder=1)
            ax.scatter(xs_spec, [yy + Y_OFFSET] * 3,
                       marker='s',
                       s=[38, 40, 38],
                       c=[C_DYS, C_POOL, C_TYP],
                       edgecolor='black', linewidth=0.6, zorder=3)

            # --- Value labels ---
            # Sensitivity: above the top sub-row
            for val, col in zip(xs_sens, (C_DYS, C_POOL, C_TYP)):
                ax.text(val, yy - LABEL_OFFSET, f"{val:.3f}",
                        ha='center', va='bottom', fontsize=6.2,
                        color=col, fontweight='bold')
            # Specificity: below the bottom sub-row
            for val, col in zip(xs_spec, (C_DYS, C_POOL, C_TYP)):
                ax.text(val, yy + LABEL_OFFSET, f"{val:.3f}",
                        ha='center', va='top', fontsize=6.2,
                        color=col)

    # ---- Block separators ----
    for i in range(1, len(y)):
        if blocks[i] != blocks[i - 1]:
            ax.axhline(i - 0.5, color='#34495E', linewidth=0.7,
                       linestyle='--', zorder=0, alpha=0.8)

    # ---- Axis cosmetics ----
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(len(methods) - 0.5, -0.7)   # inverted — first method on top
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=8, length=3)
    ax.set_xlabel('Score', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='x', linewidth=0.4)
    ax.set_axisbelow(True)

    # ---- Legend (below the axes) ----
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='Sensitivity'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markeredgecolor='black', markeredgewidth=0.6, markersize=6,
               label='Specificity'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DYS,
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='Dys'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_TYP,
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='Typ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_POOL,
               markeredgecolor='black', markeredgewidth=0.6, markersize=7,
               label='Pooled'),
    ]
    ax.legend(handles=legend_handles,
              loc='lower center', bbox_to_anchor=(0.5, -0.22),
              ncol=5, frameon=False, fontsize=7,
              handletextpad=0.3, columnspacing=1.0)

    # ---- Footnote for asterisked methods ----
    if any(flags):
        fig.text(0.5, -0.03,
                 '* subgroup data unavailable for this method.',
                 fontsize=6, color='#2C3E50', ha='center', va='top')

    plt.tight_layout()

    out_png = output_dir / f'sens_spec_compact_{task_type}.png'
    out_pdf = output_dir / f'sens_spec_compact_{task_type}.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    # ---- Sidecar CSV ----
    grouped.to_csv(
        output_dir / f'sens_spec_compact_data_{task_type}.csv', index=False)

    print(f"✓ Compact Sens/Spec plot saved: {out_png}")
    print(f"  Blocks:")
    for b in block_order:
        names = grouped[grouped['Block'] == b]['Method_Label'].tolist()
        if names:
            print(f"    {b} ({len(names)}): {names}")
    if any(flags):
        starred = [m for m, f in zip(methods, flags) if f]
        print(f"  Methods with * (subgroup data missing): {starred}")

    return grouped

def plot_subgroup_comparison(df, output_dir, config, task_type='classification'):
    plot_sens_spec_three_marker(df, output_dir, config, task_type=task_type)
    plot_sens_spec_compact(df, output_dir, config, task_type=task_type)

    metrics_to_compare = (['macro_f1', 'balanced_accuracy', 'roc_auc',
                           'sensitivity', 'specificity']
                          if task_type == 'classification' else ['rmse', 'r2'])
    for metric in metrics_to_compare:
        sm, nsm = f'subgroup_{metric}', f'non_subgroup_{metric}'
        if sm not in df.columns or nsm not in df.columns:
            continue
        plot_data = df[df['Task'] == task_type].copy()
        plot_data['Model_Short'] = plot_data['Model'].apply(
            lambda x: get_ultra_short_model_name(x, config))
        plot_data = plot_data.groupby('Model_Short').agg(
            {sm: 'mean', nsm: 'mean'}).reset_index()
        if plot_data.empty:
            continue
        lower_is_better = metric in ['rmse']
        plot_data = plot_data.sort_values(sm, ascending=lower_is_better)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_data))
        width = 0.35
        ax.bar(x - width / 2, plot_data[sm], width, label='Dys (Subgroup)',
               color='#E74C3C', alpha=0.8)
        ax.bar(x + width / 2, plot_data[nsm], width,
               label='Norm (Non-Subgroup)', color='#3498DB', alpha=0.8)
        metric_label = config.metric_labels.get(metric, metric.upper())
        ax.set_title(f'{metric_label}: Dys vs Norm ({task_type.title()})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['Model_Short'], rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        if metric not in ['rmse']:
            ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        plt.savefig(output_dir / f'subgroup_comparison_{metric}_{task_type}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_meta_fusion_comparison(df, output_dir, config, task_type='classification'):
    meta_methods = [m for m in df['Method'].unique()
                    if m.startswith('ensemble_') or m == 'meta_fusion']
    if not meta_methods:
        return
    meta_df = df[df['Method'].isin(meta_methods)].copy()
    if meta_df.empty:
        return
    meta_df['Method_Label'] = meta_df['Method'].apply(
        lambda x: get_method_display_name(x, config))
    meta_df['Model_Short'] = meta_df['Model'].apply(
        lambda x: get_ultra_short_model_name(x, config))
    meta_df = meta_df[meta_df['Task'] == task_type]
    if meta_df.empty:
        return

    metrics = (['macro_f1', 'roc_auc', 'balanced_accuracy']
               if task_type == 'classification' else ['rmse', 'r2'])

    for metric in metrics:
        if metric not in meta_df.columns:
            continue
        valid = meta_df[meta_df[metric].notna()]
        if valid.empty:
            continue
        pivot = valid.pivot_table(index='Model_Short', columns='Method_Label',
                                  values=metric, aggfunc='mean')
        if pivot.empty:
            continue
        lower_is_better = metric in ['rmse']
        best = pivot.min(axis=1) if lower_is_better else pivot.max(axis=1)
        pivot = pivot.loc[best.sort_values(ascending=lower_is_better).index]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot.index) * 0.5)))
        pivot.plot(kind='barh', ax=ax, width=0.7, colormap='viridis')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', fontsize=9, padding=2)
        metric_label = config.metric_labels.get(metric, metric.upper())
        ax.set_title(f'Meta-Fusion: {metric_label} ({task_type.title()})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(metric_label)
        ax.set_ylabel('Model')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.savefig(output_dir / f'meta_fusion_comparison_{metric}_{task_type}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_summary_table(df, output_dir, config, prefix=''):
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue
        if task == 'classification':
            metrics = [m for m in config.classification_metrics if m in task_df.columns]
            sub_metrics = [m for m in config.classification_subgroup_metrics
                           if m in task_df.columns]
        else:
            metrics = [m for m in config.regression_metrics if m in task_df.columns]
            sub_metrics = [m for m in config.regression_subgroup_metrics
                           if m in task_df.columns]
        all_metrics = metrics + sub_metrics
        agg_dict = {m: ['mean', 'std', 'count'] for m in all_metrics
                    if m in task_df.columns}
        if not agg_dict:
            continue

        task_df['Model_Short'] = task_df['Model'].apply(
            lambda x: get_ultra_short_model_name(x, config))
        group_cols = ['Model_Short']
        if 'Method_Label' in task_df.columns:
            group_cols.append('Method_Label')
        elif 'Method' in task_df.columns:
            group_cols.append('Method')

        summary = task_df.groupby(group_cols).agg(agg_dict).round(4)
        filename = (f'{prefix}summary_table_{task}.csv' if prefix
                    else f'summary_table_{task}.csv')
        summary.to_csv(output_dir / filename)
        print(f"✓ Summary table saved: {filename}")

        flat = summary.copy()
        flat.columns = [f'{c[0]}_{c[1]}' for c in flat.columns]
        flat.reset_index().to_csv(
            output_dir / (f'{prefix}summary_table_flat_{task}.csv' if prefix
                          else f'summary_table_flat_{task}.csv'), index=False)


# =======================================================================
#  MAIN
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment Results Aggregator '
                    '(JSON + prediction-derived metrics for every method)')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./results_summary')
    parser.add_argument('--task', type=str,
                        choices=['classification', 'regression', 'all'],
                        default='all')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--ignore-methods', nargs='+', default=None)
    parser.add_argument('--metrics', nargs='+', default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--dys-ids', type=str, default=None)
    parser.add_argument('--bootstrap-iterations', type=int, default=1000)
    parser.add_argument('--no-robustness', action='store_true')
    parser.add_argument('--no-scatter', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--subgroup', action='store_true')
    parser.add_argument('--include-ensembles-in-scatter', action='store_true')
    parser.add_argument('--hide-ensembles-in-plots', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig()
    config.bootstrap_iterations = args.bootstrap_iterations

    dys_ids = []
    if args.dys_ids:
        dys_ids = load_dys_speaker_ids(Path(args.dys_ids))
    if not dys_ids:
        print("⚠ No Dys speaker IDs provided. Subgroup metrics will only be "
              "computed when a prediction CSV contains a speaker-ID column; "
              "otherwise only combined metrics are produced.")

    base_dir = Path(args.input_dir)
    experiments = discover_experiments(base_dir, args.task, config)
    if not experiments:
        print("\nERROR: No experiments found!")
        return

    if args.models:
        filtered = {}
        for mn, md in experiments.items():
            for am in args.models:
                if am in mn or mn in am:
                    filtered[mn] = md
                    break
        experiments = filtered
        if not experiments:
            print("ERROR: No experiments match the specified models!")
            return

    print(f"\n{'='*60}\nAGGREGATING RESULTS (JSON + predictions)\n{'='*60}")
    df = aggregate_experiment_results(experiments, config, dys_ids)
    print(f"Aggregated {len(df)} method results")

    if args.ignore_methods:
        ignore_set = set(args.ignore_methods)
        df = df[~df['Method'].isin(ignore_set) & ~df['Method_Label'].isin(ignore_set)]

    if args.top_k and args.top_k > 0:
        df = select_top_k_models(df, args.top_k,
                                 args.task if args.task != 'all' else None,
                                 config)

    if args.task != 'all':
        df = df[df['Task'] == args.task]
    if args.models:
        df = df[df['Model'].isin(args.models)]
    if args.methods:
        df = df[df['Method'].isin(args.methods)]

    if df.empty:
        print("ERROR: No data found after filtering!")
        return

    print("\nTasks:", df['Task'].unique().tolist())
    print("Models:", df['Model'].unique().tolist())
    print("Methods:", df['Method_Label'].unique().tolist())

    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"✓ all_results.csv written ({len(df)} rows)")

    print_method_coverage(df)

    if 'regression' in df['Task'].unique():
        print_regression_sanity_check(df, config)
        diagnose_diverged_predictions(experiments, output_dir, config)

    if 'classification' in df['Task'].unique():
        print_classification_sanity_check(df, config)

    print(f"\n{'='*60}\nGENERATING SUMMARY TABLES\n{'='*60}")
    create_summary_table(df, output_dir, config)

    if not args.no_robustness:
        print(f"\n{'='*60}\nROBUSTNESS ANALYSIS\n{'='*60}")
        for task in df['Task'].unique():
            plot_robustness_summary_all_patients(df, output_dir, config, task,
                                                 args.bootstrap_iterations)

    if not args.no_scatter and 'regression' in df['Task'].unique():
        print(f"\n{'='*60}\nDYS + NORM SCATTER PLOTS\n{'='*60}")
        if args.include_ensembles_in_scatter:
            scatter_df = df
        else:
            scatter_df = filter_out_ensembles(df, 'Method', 'Method_Label')
        plot_dys_scatter_audio_text_fusion_single(scatter_df, experiments,
                                                  output_dir, config,
                                                  'regression', dys_ids)

    if not args.no_plots:
        print(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")
        for task in df['Task'].unique():
            if task == 'classification':
                default_metrics = ['macro_f1', 'sensitivity', 'specificity',
                                   'balanced_accuracy', 'roc_auc']
            else:
                default_metrics = ['rmse', 'r2']

            metrics_to_plot = (args.metrics if args.metrics
                               else [m for m in default_metrics
                                     if m in df.columns
                                     and df[df['Task'] == task][m].notna().any()])

            for metric in metrics_to_plot:
                if (metric in df.columns
                        and df[df['Task'] == task][metric].notna().any()):
                    plot_heatmap(df, metric, output_dir, config,
                                 task_type=task, hide_ensembles=False)
                    plot_heatmap(df, metric, output_dir, config,
                                 task_type=task, hide_ensembles=True)
                    plot_method_comparison(
                        df, metric, output_dir, config, task_type=task,
                        hide_ensembles=args.hide_ensembles_in_plots)

            if args.subgroup:
                plot_subgroup_comparison(df, output_dir, config, task_type=task)

        has_meta = any(df['Method'].str.startswith('ensemble_')
                       | (df['Method'] == 'meta_fusion'))
        if has_meta:
            for task in df['Task'].unique():
                plot_meta_fusion_comparison(df, output_dir, config, task_type=task)

    print(f"\n{'='*60}\nSUMMARY STATISTICS\n{'='*60}")
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        print(f"\n{task.upper()}:")

        if task == 'classification':
            ranking_metric = config.ranking_metric_classification
            lower_is_better = False
        else:
            ranking_metric = config.ranking_metric_regression
            lower_is_better = True

        if ranking_metric not in task_df.columns:
            continue

        metric_label = config.metric_labels.get(ranking_metric, ranking_metric.upper())
        print(f"\n  Ranking metric: {metric_label} "
              f"({'lower' if lower_is_better else 'higher'} is better)")

        print(f"\n  Best by Method (mean):")
        best_by_method = task_df.groupby('Method_Label')[ranking_metric].mean()
        best_by_method = best_by_method.sort_values(ascending=lower_is_better)
        for method, val in best_by_method.head(5).items():
            print(f"    {method}: {val:.4f}")

        print(f"\n  Best by Model:")
        best_by_model = task_df.groupby('Model')[ranking_metric].mean()
        best_by_model = best_by_model.sort_values(ascending=lower_is_better)
        for model, val in best_by_model.head(5).items():
            print(f"    {get_ultra_short_model_name(model, config)}: {val:.4f}")

    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

'''

for t in classification regression;do rm -rf outputs-ensemble-aggregate-$t;mkdir outputs-ensemble-aggregate-$t;python ~/asr_clinical/question_ensemble_fusion_aggregate_results.py --input-dir outputs-ensemble --output-dir outputs-ensemble-aggregate-$t --subgroup --top-k 5 --task $t --bootstrap-iterations 10000 --verbose  --dys-ids dysarthria-list.txt --ignore-methods mlp cca dynamic ensemble_average ensemble_weighted | tee outputs-ensemble-aggregate-$t/log.txt;done

'''