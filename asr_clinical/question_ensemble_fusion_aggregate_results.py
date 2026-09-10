"""
Experiment Results Aggregator - Enhanced Version
Supports BOTH classification and regression experiments with subgroup analysis (Dys/Norm).

Key Metrics:
- Classification: F1-Macro, Sensitivity, Specificity, Balanced Accuracy, AUC-ROC
- Regression: RMSE, R²
- Subgroup analysis: Dys vs Norm for all metrics with Sen/Spec pairs
- Robustness: Confidence Intervals (95%), Bootstrap resampling (1000 iterations)
- Ablation: Component removal analysis using ALL patients data (not subgroups)
- Scatter plots: Audio-Only vs Best Text vs Best Fusion for Dys subgroup

Folder Structure:
<main_dir>/
├── classification-fusion-<model_name>/           
│   ├── fusion_results/
│   │   ├── leakage_safe_5fold/
│   │   │   ├── audio_only_aggregate_metrics.json
│   │   │   ├── text_only_aggregate_metrics.json
│   │   │   ├── fuse-*_aggregate_metrics.json
│   │   │   └── ...
│   │   └── meta_fusion/
│   │       ├── meta_fusion_metrics.json
│   │       ├── stacked_predictions.csv
│   │       └── ...
├── regression-fusion-<model_name>/               
│   └── ...
└── ...
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
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
        sns.set_style("whitegrid")

sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# =======================================================================
#  CONFIGURATION
# =======================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the aggregator."""
    
    # All 14 fusion methods and their JSON filenames
    fusion_methods: Dict[str, str] = field(default_factory=lambda: {
        # BASE METHODS (4)
        'audio_only': 'audio_only_aggregate_metrics.json',
        'text_only': 'text_only_aggregate_metrics.json',
        'early': 'early_aggregate_metrics.json',
        'late': 'late_aggregate_metrics.json',
        # ADVANCED METHODS (10)
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
    
    # Display names for all 14 fusion methods
    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        # BASE METHODS (4)
        'audio_only': 'Audio-Only',
        'text_only': 'Text-Only',
        'early': 'Early Fusion',
        'late': 'Late Fusion',
        # ADVANCED METHODS (10)
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
    
    # Display names for fusion combinations (fuse- prefix)
    fusion_combination_display_names: Dict[str, str] = field(default_factory=lambda: {
        'fuse-audio_text': 'Audio + Text',
        'fuse-audio_early': 'Audio + Early',
        'fuse-text_early': 'Text + Early',
        'fuse-audio_text_early': 'Audio + Text + Early',
        'fuse-early_late': 'Early + Late',
        'fuse-audio_text_late': 'Audio + Text + Late',
        'fuse-confidence_audio': 'Confidence + Audio',
        'fuse-confidence_text': 'Confidence + Text',
        'fuse-confidence_early': 'Confidence + Early',
        'fuse-moe_stacking': 'MoE + Stacking',
        'fuse-cca_mlp': 'CCA + MLP',
        'fuse-dynamic_confidence': 'Dynamic + Confidence',
        # NEW: Advanced fusion combinations
        'fuse-cross_attention_bilinear': 'Cross-Attention + Bilinear',
        'fuse-adaptive_weighted_confidence': 'Adaptive + Confidence',
        'fuse-mlp_cross_attention': 'MLP + Cross-Attention',
    })
    
    # Meta-fusion display names (5 strategies)
    meta_fusion_display_names: Dict[str, str] = field(default_factory=lambda: {
        'average': 'Average Ensemble',
        'voting': 'Voting Ensemble',
        'stacking': 'Stacking Ensemble',
        'weighted': 'Weighted Ensemble',
        'confidence_selection': 'Confidence Selection',
        'best': 'Best Method'
    })
    
    # Model name mapping for shorter display names
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
    
    # Classification metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        'macro_f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'roc_auc'
    ])
    
    # Regression metrics
    regression_metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'r2'
    ])
    
    # Subgroup metrics (for classification)
    classification_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_macro_f1', 'subgroup_sensitivity', 'subgroup_specificity', 
        'subgroup_balanced_accuracy', 'subgroup_roc_auc',
        'non_subgroup_macro_f1', 'non_subgroup_sensitivity', 'non_subgroup_specificity', 
        'non_subgroup_balanced_accuracy', 'non_subgroup_roc_auc'
    ])
    
    # Subgroup metrics (for regression)
    regression_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_rmse', 'subgroup_r2',
        'non_subgroup_rmse', 'non_subgroup_r2'
    ])
    
    # Metric labels
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
    
    # Ranking metrics
    ranking_metric_classification: str = 'macro_f1'
    ranking_metric_regression: str = 'r2'
    
    # Bootstrap iterations
    bootstrap_iterations: int = 1000

    # Add this to ExperimentConfig:
    fusion_method_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'Base Methods': ['audio_only', 'text_only', 'early', 'late'],
        'Advanced Methods': ['confidence', 'interaction', 'moe', 'mlp', 'stacking', 'cca', 'dynamic'],
        'State-of-the-Art': ['cross_attention', 'adaptive_weighted', 'bilinear']
    })

    # Method colors for consistent visualization
    method_colors: Dict[str, str] = field(default_factory=lambda: {
        # Base Methods
        'audio_only': '#E74C3C',
        'text_only': '#3498DB',
        'early': '#2ECC71',
        'late': '#F1C40F',
        # Advanced Methods
        'confidence': '#9B59B6',
        'interaction': '#1ABC9C',
        'moe': '#E67E22',
        'mlp': '#3498DB',
        'stacking': '#2C3E50',
        'cca': '#8E44AD',
        'dynamic': '#16A085',
        # State-of-the-Art
        'cross_attention': '#E74C3C',
        'adaptive_weighted': '#27AE60',
        'bilinear': '#2980B9'
    })


# =======================================================================
#  HELPER FUNCTIONS
# =======================================================================

def get_short_model_name(model_name: str, config: ExperimentConfig) -> str:
    """Get shortened display name for a model."""
    if model_name in config.model_name_mapping:
        return config.model_name_mapping[model_name]
    
    for key, value in config.model_name_mapping.items():
        if key in model_name or model_name in key:
            return value
    
    cleaned = model_name
    prefixes = ['microsoft.', 'google.', 'facebook/', 'allenai/', 'emilyalsentzer/']
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    
    suffixes = ['-uncased', '-cased', '-v2', '-base', '-large', '-small']
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    
    return cleaned.title()


def get_method_display_name(method_key: str, config: ExperimentConfig) -> str:
    """Get display name for a method, handling all 14 methods + fuse- combinations."""
    # Handle standard fusion methods
    if method_key in config.method_display_names:
        return config.method_display_names[method_key]
    
    # Handle fuse- combinations
    if method_key.startswith('fuse-'):
        if method_key in config.fusion_combination_display_names:
            return config.fusion_combination_display_names[method_key]
        else:
            parts = method_key.replace('fuse-', '').split('_')
            return ' + '.join([p.replace('_', ' ').title() for p in parts])
    
    # Handle ensemble/meta-fusion methods
    if method_key.startswith('ensemble_'):
        ensemble = method_key.replace('ensemble_', '')
        return config.meta_fusion_display_names.get(ensemble, ensemble.title())
    
    # Handle meta_fusion directly
    if method_key == 'meta_fusion':
        return 'Meta-Fusion'
    
    # Fallback
    return method_key.replace('_', ' ').title()


def clean_model_name(model_name: str) -> str:
    """Clean model name by removing common prefixes."""
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
    """Detect if model is base or large."""
    model_lower = model_name.lower()
    if 'large' in model_lower:
        return 'Large'
    elif 'base' in model_lower:
        return 'Base'
    elif 'small' in model_lower:
        return 'Small'
    else:
        return 'Unknown'


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    model_lower = model_name.lower()
    if 'roberta' in model_lower:
        return 'RoBERTa'
    elif 'bert' in model_lower:
        return 'BERT'
    elif 'distil' in model_lower:
        return 'DistilRoBERTa'
    elif 'albert' in model_lower:
        return 'ALBERT'
    elif 'deberta' in model_lower:
        return 'DeBERTa'
    elif 'clinical' in model_lower:
        return 'Clinical BERT'
    elif 'biomed' in model_lower:
        return 'BioMed'
    elif 'scibert' in model_lower:
        return 'SciBERT'
    elif 'legal' in model_lower:
        return 'Legal BERT'
    else:
        return 'Other'


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """Parse folder name."""
    result = {
        'task': 'unknown',
        'model': folder_name,
        'full_name': folder_name,
        'size': 'Unknown',
        'family': 'Other'
    }
    
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
#  SUBGROUP MAPPING FUNCTIONS
# =======================================================================

def load_dys_speaker_ids(mapping_file: Path) -> List[str]:
    """
    Load Dys speaker IDs from a simple text file (one ID per line).
    Returns a list of speaker IDs that belong to the Dys subgroup.
    """
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


def filter_predictions_by_dys_ids(pred_df: pd.DataFrame, dys_ids: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Filter predictions DataFrame to only include speakers from the Dys subgroup.
    Deduplicates by speaker ID to handle cross-validation folds.
    """
    if not dys_ids:
        return pred_df, len(pred_df)
    
    # Try to find speaker ID column in predictions
    id_col = None
    for col in pred_df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['speaker', 'participant', 'subject', 'patient', 'id']):
            id_col = col
            break
    
    if id_col is None:
        print(f"  Warning: No speaker ID column found in predictions")
        print(f"  Available columns: {pred_df.columns.tolist()}")
        return pred_df, len(pred_df)
    
    # Convert IDs to string for matching
    pred_ids = pred_df[id_col].astype(str)
    dys_ids_str = [str(id_val) for id_val in dys_ids]
    
    # Filter to Dys IDs only
    filtered_df = pred_df[pred_ids.isin(dys_ids_str)]
    
    # Deduplicate by ID to get unique speakers (remove cross-validation folds)
    if len(filtered_df) > 0 and id_col in filtered_df.columns:
        filtered_df = filtered_df.drop_duplicates(subset=[id_col], keep='first')
    
    n_samples = len(filtered_df)
    print(f"  Filtered to {n_samples} unique Dys samples (from {len(dys_ids)} Dys IDs)")
    
    return filtered_df, n_samples


# =======================================================================
#  METRIC LOADING FUNCTIONS
# =======================================================================

def load_metrics_file(file_path: Path) -> Optional[Dict]:
    """Load metrics from JSON file."""
    if not file_path or not file_path.exists():
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def extract_metrics_from_result(result: Dict, task: str) -> Dict:
    """Extract metrics from result dictionary."""
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
                if key not in ['threshold', 'threshold_used', 'k_neighbors', 'avg_best_k']:
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
    """Load predictions from CSV file."""
    if not file_path or not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        return None


def load_subgroup_predictions(experiments: Dict, model_name: str, method_key: str,
                              dys_ids: List[str] = None) -> Optional[pd.DataFrame]:
    """
    Load predictions for a specific model and method.
    Filters for Dys subgroup if dys_ids are provided.
    """
    if model_name not in experiments:
        return None
    
    model_data = experiments[model_name]
    
    # Try to find prediction file
    pred_key = f'predictions_{method_key}'
    if pred_key in model_data:
        pred_file = model_data[pred_key].get('predictions_file')
        if pred_file and pred_file.exists():
            df = load_predictions_file(pred_file)
            if df is not None and dys_ids:
                df, n = filter_predictions_by_dys_ids(df, dys_ids)
            return df
    
    # Try meta_fusion directory
    meta_fusion_dir = Path(model_data.get('meta_fusion', {}).get('dir', '')) / 'fusion_results' / 'meta_fusion'
    if meta_fusion_dir.exists():
        pred_files = list(meta_fusion_dir.glob(f"*{method_key}*_predictions.csv"))
        pred_files.extend(meta_fusion_dir.glob(f"*{method_key}*_predictions.txt"))
        for pred_file in pred_files:
            df = load_predictions_file(pred_file)
            if df is not None:
                if dys_ids:
                    df, n = filter_predictions_by_dys_ids(df, dys_ids)
                return df
    
    # Try leakage_safe_5fold directory
    leakage_dir = Path(model_data.get('audio_only', {}).get('dir', '')) / 'fusion_results' / 'leakage_safe_5fold'
    if leakage_dir.exists():
        pred_files = list(leakage_dir.glob(f"*{method_key}*_predictions.csv"))
        for pred_file in pred_files:
            df = load_predictions_file(pred_file)
            if df is not None:
                if dys_ids:
                    df, n = filter_predictions_by_dys_ids(df, dys_ids)
                return df
    
    return None


# =======================================================================
#  EXPERIMENT DISCOVERY
# =======================================================================

def discover_experiments(base_dir: Path, task_type: str = 'all', config: ExperimentConfig = None) -> Dict:
    """Discover experiments by scanning folders - supports all 14 fusion methods + meta-fusion."""
    if config is None:
        config = ExperimentConfig()
    
    experiments = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n{'='*60}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*60}")
    print(f"Task type filter: {task_type}")
    print(f"Looking for {len(config.fusion_methods)} fusion methods + meta-fusion")
    
    # Find all folders
    all_folders = []
    
    if task_type in ['classification', 'all']:
        classification_folders = list(base_dir.glob("classification*"))
        all_folders.extend(classification_folders)
        print(f"Found {len(classification_folders)} classification folders")
    
    if task_type in ['regression', 'all']:
        regression_folders = list(base_dir.glob("regression*"))
        all_folders.extend(regression_folders)
        print(f"Found {len(regression_folders)} regression folders")
    
    all_folders = list(set(all_folders))
    print(f"\nTotal: {len(all_folders)} experiment folders to process")
    
    # Track which methods were found
    found_methods = set()
    
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
        
        print(f"\nProcessing: {folder_name}")
        print(f"  Task: {task}")
        print(f"  Model: {model_name}")
        print(f"  Size: {size}, Family: {family}")
        
        leakage_dir = folder_path / 'fusion_results' / 'leakage_safe_5fold'
        meta_fusion_dir = folder_path / 'fusion_results' / 'meta_fusion'
        
        print(f"  Checking leakage_safe_5fold: {leakage_dir}")
        
        found_fusion = False
        
        if leakage_dir.exists():
            # Look for ALL JSON files in the directory
            json_files = list(leakage_dir.glob("*_aggregate_metrics.json"))
            json_files.extend(leakage_dir.glob("*_metrics.json"))
            json_files = list(set(json_files))
            
            print(f"    Found {len(json_files)} metric files in leakage_safe_5fold")
            
            for json_file in json_files:
                filename = json_file.name
                method_key = None
                
                # Check standard fusion methods
                for key, fname in config.fusion_methods.items():
                    if filename == fname:
                        method_key = key
                        break
                
                # If not standard, check if it's a fuse- combination
                if method_key is None:
                    base_name = filename
                    for suffix in ['_aggregate_metrics.json', '_metrics.json', '.json']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    if base_name.startswith('fuse-'):
                        method_key = base_name
                    else:
                        # Try to match with known methods
                        for key in config.fusion_methods.keys():
                            if key in base_name or base_name in key:
                                method_key = key
                                break
                        
                        if method_key is None:
                            method_key = base_name
                
                # Store the result
                experiments[model_name][method_key] = {
                    'task': task,
                    'model': model_name,
                    'size': size,
                    'family': family,
                    'metrics_file': json_file,
                    'dir': folder_path,
                    'source': 'leakage_safe_5fold'
                }
                found_fusion = True
                found_methods.add(method_key)
                print(f"    ✓ Found: {method_key} -> {json_file.name}")
        else:
            print(f"    ✗ leakage_safe_5fold NOT found")
            fusion_results_dir = folder_path / 'fusion_results'
            if fusion_results_dir.exists():
                subdirs = list(fusion_results_dir.glob("*"))
                print(f"    Available in fusion_results: {[s.name for s in subdirs]}")
        
        # Check for meta_fusion results
        if meta_fusion_dir.exists():
            meta_fusion_file = meta_fusion_dir / 'meta_fusion_metrics.json'
            
            if meta_fusion_file.exists():
                meta_metrics = load_metrics_file(meta_fusion_file)
                if meta_metrics:
                    # Extract ensemble methods from meta_fusion
                    ensemble_methods = ['average', 'voting', 'stacking', 'weighted', 'confidence_selection', 'best']
                    for ensemble in ensemble_methods:
                        if ensemble in meta_metrics and isinstance(meta_metrics[ensemble], dict):
                            ensemble_key = f'ensemble_{ensemble}'
                            experiments[model_name][ensemble_key] = {
                                'task': task,
                                'model': model_name,
                                'size': size,
                                'family': family,
                                'metrics_file': meta_fusion_file,
                                'dir': folder_path,
                                'source': 'meta_fusion',
                                'ensemble_method': ensemble
                            }
                            print(f"    ✓ Found ensemble: {ensemble}")
                            
                    # Also store the full meta_fusion results
                    experiments[model_name]['meta_fusion'] = {
                        'task': task,
                        'model': model_name,
                        'size': size,
                        'family': family,
                        'metrics_file': meta_fusion_file,
                        'dir': folder_path,
                        'source': 'meta_fusion'
                    }
                    print(f"    ✓ Found meta_fusion_metrics.json")
            
            # Look for prediction files
            pred_files = list(meta_fusion_dir.glob("*_predictions.csv"))
            pred_files.extend(meta_fusion_dir.glob("*_predictions.txt"))
            for pred_file in pred_files:
                experiments[model_name][f'predictions_{pred_file.stem}'] = {
                    'task': task,
                    'model': model_name,
                    'size': size,
                    'family': family,
                    'predictions_file': pred_file,
                    'dir': folder_path,
                    'source': 'predictions'
                }
                print(f"    ✓ Found predictions: {pred_file.name}")
        else:
            print(f"    ✗ meta_fusion NOT found")
        
        if not found_fusion:
            print(f"    ⚠ No fusion metrics found in leakage_safe_5fold")
    
    # Print summary with method counts
    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    
    # Count methods found
    method_counts = defaultdict(int)
    for model_data in experiments.values():
        for method_key in model_data.keys():
            if method_key not in ['meta_fusion', 'meta_fusion_summary'] and not method_key.startswith('predictions_'):
                method_counts[method_key] += 1
    
    print(f"\nMethods found across all models:")
    # Group by category
    for category, methods in config.fusion_method_groups.items():
        found_in_category = [m for m in methods if m in method_counts]
        if found_in_category:
            print(f"  {category}: {len(found_in_category)}/{len(methods)} methods")
            for m in found_in_category:
                print(f"    ✓ {get_method_display_name(m, config)}: {method_counts[m]} models")
    
    # Check for meta-fusion
    meta_fusion_count = sum(1 for model_data in experiments.values() if 'meta_fusion' in model_data)
    if meta_fusion_count > 0:
        print(f"\n  Meta-Fusion: found in {meta_fusion_count} models")
    
    total_methods = sum(method_counts.values())
    print(f"\nTotal: {len(experiments)} models, {total_methods} fusion method results")
    
    return dict(experiments)

def plot_meta_fusion_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                                 task_type: str = 'classification'):
    """Create visualization comparing meta-fusion methods."""
    # Filter for meta-fusion methods
    meta_methods = [m for m in df['Method'].unique() if m.startswith('ensemble_') or m == 'meta_fusion']
    if not meta_methods:
        print("No meta-fusion methods found")
        return
    
    meta_df = df[df['Method'].isin(meta_methods)].copy()
    
    if meta_df.empty:
        print("No meta-fusion data found")
        return
    
    # Get display names
    meta_df['Method_Label'] = meta_df['Method'].apply(
        lambda x: get_method_display_name(x, config)
    )
    
    # Add Model_Short column - THIS WAS MISSING!
    meta_df['Model_Short'] = meta_df['Model'].apply(
        lambda x: get_short_model_name(x, config)
    )
    
    # Filter to task
    meta_df = meta_df[meta_df['Task'] == task_type]
    
    if meta_df.empty:
        print(f"No meta-fusion data for task {task_type}")
        return
    
    # Get metrics based on task
    if task_type == 'classification':
        metrics = ['macro_f1', 'roc_auc', 'balanced_accuracy']
    else:
        metrics = ['rmse', 'r2']
    
    # Create comparison plots
    for metric in metrics:
        if metric not in meta_df.columns:
            continue
        
        # Check if we have valid data
        valid_data = meta_df[meta_df[metric].notna()]
        if valid_data.empty:
            print(f"No valid {metric} data for meta-fusion comparison")
            continue
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(valid_data['Model_Short'].unique()) * 0.5)))
        
        # Pivot table
        pivot = valid_data.pivot_table(
            index='Model_Short',
            columns='Method_Label',
            values=metric,
            aggfunc='mean'
        )
        
        if pivot.empty:
            print(f"Empty pivot for {metric}")
            continue
        
        # Sort by best method
        if metric in ['rmse']:
            best_vals = pivot.min(axis=1)
            pivot = pivot.loc[best_vals.sort_values(ascending=True).index]
        else:
            best_vals = pivot.max(axis=1)
            pivot = pivot.loc[best_vals.sort_values(ascending=False).index]
        
        # Plot
        pivot.plot(kind='barh', ax=ax, width=0.7, colormap='viridis')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9, padding=2)
        
        metric_label = config.metric_labels.get(metric, metric.upper())
        ax.set_title(f'Meta-Fusion Comparison: {metric_label} ({task_type.title()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        
        if metric not in ['rmse']:
            ax.set_xlim(0, 1.05)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        task_suffix = f"_{task_type}"
        plt.savefig(output_dir / f'meta_fusion_comparison_{metric}{task_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Meta-fusion comparison saved to: {output_dir / f'meta_fusion_comparison_{metric}{task_suffix}.png'}")

# =======================================================================
#  DATA AGGREGATION FUNCTIONS
# =======================================================================

def aggregate_experiment_results(experiments: Dict, config: ExperimentConfig) -> pd.DataFrame:
    """Aggregate all experiment results into a single DataFrame."""
    rows = []
    
    for model_name, model_data in experiments.items():
        task = 'unknown'
        size = 'Unknown'
        family = 'Other'
        
        for method_key, method_data in model_data.items():
            if 'task' in method_data:
                task = method_data.get('task', 'unknown')
                size = method_data.get('size', 'Unknown')
                family = method_data.get('family', 'Other')
                break
        
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
                    extracted = extract_metrics_from_result(metrics[ensemble_method], task)
                    display_name = config.meta_fusion_display_names.get(ensemble_method, ensemble_method.title())
                    method_display = display_name
                    source = 'meta_fusion'
                else:
                    continue
            else:
                metrics = load_metrics_file(metrics_file)
                if metrics is None:
                    continue
                
                extracted = extract_metrics_from_result(metrics, task)
                method_display = get_method_display_name(method_key, config)
                source = method_data.get('source', 'leakage_safe_5fold')
            
            row = {
                'Task': task,
                'Model': model_name,
                'Size': size,
                'Family': family,
                'Method': method_key,
                'Method_Label': method_display,
                'Source': source
            }
            
            if task == 'classification':
                all_metrics = config.classification_metrics + config.classification_subgroup_metrics
            else:
                all_metrics = config.regression_metrics + config.regression_subgroup_metrics
            
            for metric in all_metrics:
                row[metric] = extracted.get(metric, None)
                if row[metric] is None and metric in metrics:
                    row[metric] = metrics.get(metric, None)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    all_metrics = (config.classification_metrics + config.classification_subgroup_metrics + 
                   config.regression_metrics + config.regression_subgroup_metrics)
    for metric in all_metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


def select_top_k_models(df: pd.DataFrame, k: int, task_type: str = None, config: ExperimentConfig = None) -> pd.DataFrame:
    """Select top K models based on the ranking metric for each task."""
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
        
        if lower_is_better:
            model_performance = model_performance.sort_values(ranking_metric, ascending=True)
        else:
            model_performance = model_performance.sort_values(ranking_metric, ascending=False)
        
        top_models = model_performance.head(k)['Model'].tolist()
        selected_models.extend(top_models)
        
        print(f"\n  Top {k} models for {task} (by {config.metric_labels.get(ranking_metric, ranking_metric)}):")
        for idx, row in model_performance.head(k).iterrows():
            print(f"    {row['Model']}: {row[ranking_metric]:.4f}")
    
    if selected_models:
        filtered_df = df[df['Model'].isin(selected_models)]
        return filtered_df
    
    return df


# =======================================================================
#  SCATTER PLOT FUNCTIONS
# =======================================================================

def plot_dys_scatter_audio_text_fusion_single(df: pd.DataFrame, experiments: Dict, output_dir: Path, 
                                                config: ExperimentConfig, task_type: str = 'regression',
                                                dys_ids: List[str] = None):
    """
    Create a single figure with three scatter plots for Dys subgroup showing:
    - Audio-Only: ECAS Observed vs ECAS Predicted (using the only audio model)
    - Best Text-Only: ECAS Observed vs ECAS Predicted (using the best text model by RMSE)
    - Best Fusion: ECAS Observed vs ECAS Predicted (using the best fusion model by RMSE)
    """
    if task_type != 'regression':
        print("Scatter plots only available for regression tasks")
        return
    
    plot_df = df[df['Task'] == 'regression'].copy()
    if plot_df.empty:
        print("No regression data found")
        return
    
    if dys_ids:
        print(f"\nUsing Dys subgroup with {len(dys_ids)} speaker IDs")
    else:
        print("\n⚠ No Dys speaker IDs provided. Using all data (not filtering).")
    
    models = plot_df['Model'].unique()
    print(f"\nFound {len(models)} models for regression")
    
    # ===== Find the BEST text-only model (by Dys RMSE) =====
    best_text_model = None
    best_text_method = None
    best_text_rmse = float('inf')
    best_text_r2 = -float('inf')
    
    # ===== Find the BEST fusion model (by Dys RMSE) =====
    best_fusion_model = None
    best_fusion_method = None
    best_fusion_rmse = float('inf')
    best_fusion_r2 = -float('inf')
    
    for model in models:
        model_df = plot_df[plot_df['Model'] == model]
        
        # Find best text-only method for this model
        text_methods = [m for m in model_df['Method'].unique() if 'text' in m.lower()]
        for method in text_methods:
            method_df = model_df[model_df['Method'] == method]
            if 'subgroup_rmse' in method_df.columns and method_df['subgroup_rmse'].notna().any():
                rmse_val = method_df['subgroup_rmse'].mean()
                r2_val = method_df['subgroup_r2'].mean() if 'subgroup_r2' in method_df.columns else -float('inf')
                if rmse_val < best_text_rmse:
                    best_text_rmse = rmse_val
                    best_text_r2 = r2_val
                    best_text_model = model
                    best_text_method = method
        
        # Find best fusion method for this model (not audio_only, not text_only)
        fusion_methods = [m for m in model_df['Method'].unique() 
                         if m not in ['audio_only'] and not m.startswith('ensemble_') 
                         and 'text' not in m.lower()]
        for method in fusion_methods:
            method_df = model_df[model_df['Method'] == method]
            if 'subgroup_rmse' in method_df.columns and method_df['subgroup_rmse'].notna().any():
                rmse_val = method_df['subgroup_rmse'].mean()
                r2_val = method_df['subgroup_r2'].mean() if 'subgroup_r2' in method_df.columns else -float('inf')
                if rmse_val < best_fusion_rmse:
                    best_fusion_rmse = rmse_val
                    best_fusion_r2 = r2_val
                    best_fusion_model = model
                    best_fusion_method = method
    
    print(f"\nBest Text-Only: {best_text_model} - {best_text_method} (RMSE={best_text_rmse:.3f}, R²={best_text_r2:.3f})")
    print(f"Best Fusion: {best_fusion_model} - {best_fusion_method} (RMSE={best_fusion_rmse:.3f}, R²={best_fusion_r2:.3f})")
    
    # Audio-Only: just use 'audio_only'
    audio_method = 'audio_only'
    audio_model = None
    for model in models:
        model_df = plot_df[plot_df['Model'] == model]
        if audio_method in model_df['Method'].unique():
            audio_model = model
            break
    
    # Create a single figure with 3 subplots - larger figure for better readability
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Increase font sizes globally for this figure
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,   # Bigger legend font
    })
    
    method_types = [
        ('audio_only', 'Audio-Only', axes[0], '#E74C3C', 'o', audio_model, audio_method),
        ('text_only', 'Best Text-Only', axes[1], '#3498DB', 's', best_text_model, best_text_method),
        ('fusion', 'Best Fusion', axes[2], '#2ECC71', '^', best_fusion_model, best_fusion_method)
    ]
    
    all_predictions = {}
    
    for method_type, display_name, ax, color, marker, model, method in method_types:
        if model is None or method is None:
            print(f"\n⚠ No model found for {display_name}")
            ax.text(0.5, 0.5, f'No model found\nfor {display_name}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=18)
            ax.set_xlabel('ECAS Observed', fontsize=18)
            ax.set_ylabel('ECAS Predicted', fontsize=18)
            ax.set_title(f'{display_name}\nDys Subgroup', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue
        
        print(f"\n{'='*50}")
        print(f"Collecting {display_name} predictions...")
        print(f"  Model: {get_short_model_name(model, config)} ({model})")
        print(f"  Method: {method}")
        print(f"{'='*50}")
        
        # Try to load predictions for this specific model and method
        pred_df = load_subgroup_predictions(experiments, model, method, dys_ids)
        
        if pred_df is not None:
            print(f"  Found predictions with {len(pred_df)} Dys samples")
            
            # Find observed, predicted, and speaker ID columns
            obs_col = None
            pred_col = None
            id_col = None
            
            for col in pred_df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['observed', 'true', 'actual', 'target', 'y_true']):
                    obs_col = col
                if any(x in col_lower for x in ['predicted', 'pred', 'y_pred', 'output']):
                    pred_col = col
                if any(x in col_lower for x in ['speaker', 'participant', 'subject', 'patient', 'id']):
                    id_col = col
            
            if obs_col is not None and pred_col is not None:
                # Get data
                if id_col is not None:
                    # Use speaker ID to deduplicate (in case of multiple folds)
                    pred_df = pred_df.drop_duplicates(subset=[id_col], keep='first')
                    obs_vals = pred_df[obs_col].values
                    pred_vals = pred_df[pred_col].values
                else:
                    obs_vals = pred_df[obs_col].values
                    pred_vals = pred_df[pred_col].values
                
                # Remove NaN values
                mask = ~(np.isnan(obs_vals) | np.isnan(pred_vals))
                obs_vals = obs_vals[mask]
                pred_vals = pred_vals[mask]
                
                if len(obs_vals) > 0:
                    print(f"  ✅ Extracted {len(obs_vals)} Dys samples")
                    
                    # Calculate metrics
                    rmse = np.sqrt(np.mean((obs_vals - pred_vals) ** 2))
                    r2 = pearsonr(obs_vals, pred_vals)[0] ** 2 if len(obs_vals) > 2 else np.nan
                    n = len(obs_vals)
                    
                    # Scatter plot - BIGGER markers with THICKER edges
                    ax.scatter(obs_vals, pred_vals, alpha=0.75, s=150, color=color, marker=marker, 
                              edgecolors='black', linewidths=1.5,
                              label=f'n={n}, RMSE={rmse:.3f}, R²={r2:.3f}')
                    
                    # Get data range for auto-scaling
                    data_min = min(obs_vals.min(), pred_vals.min())
                    data_max = max(obs_vals.max(), pred_vals.max())
                    data_range = data_max - data_min
                    
                    # Add some padding (5% of range)
                    padding = data_range * 0.05 if data_range > 0 else 1.0
                    plot_min = data_min - padding
                    plot_max = data_max + padding
                    
                    # Add identity line (y=x) - THICKER
                    ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                           'k--', alpha=0.6, linewidth=3, label='y = x (perfect)')
                    
                    # Add regression line - THICKER
                    if len(obs_vals) > 2:
                        slope, intercept, r_value, p_value, std_err = linregress(obs_vals, pred_vals)
                        x_line = np.linspace(plot_min, plot_max, 100)
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color='red', alpha=0.8, linewidth=3, 
                               label=f'y={slope:.2f}x+{intercept:.2f}')
                    
                    ax.set_xlabel('ECAS Observed', fontsize=18)
                    ax.set_ylabel('ECAS Predicted', fontsize=18)
                    ax.set_title(f'{display_name}\nDys Subgroup', fontsize=20, fontweight='bold')
                    # BIGGER legend font
                    ax.legend(loc='best', fontsize=15, framealpha=0.95, 
                              handletextpad=0.5, borderpad=0.6, labelspacing=0.6)
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Set limits from min to max with padding (NOT 0 to 136)
                    ax.set_xlim(plot_min, plot_max)
                    ax.set_ylim(plot_min, plot_max)
                    
                    # Store predictions
                    all_predictions[method_type] = {
                        'model': model,
                        'method': method,
                        'obs': obs_vals,
                        'pred': pred_vals,
                        'rmse': rmse,
                        'r2': r2,
                        'n': n
                    }
                else:
                    print(f"  ⚠ No valid samples found")
                    ax.text(0.5, 0.5, f'No valid samples', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=18)
            else:
                print(f"  ⚠ Could not find obs/pred columns")
                print(f"  Available columns: {pred_df.columns.tolist()}")
                ax.text(0.5, 0.5, f'No obs/pred columns', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=18)
        else:
            print(f"  ⚠ No predictions file found")
            ax.text(0.5, 0.5, f'No predictions file', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=18)
        
        ax.set_xlabel('ECAS Observed', fontsize=18)
        ax.set_ylabel('ECAS Predicted', fontsize=18)
        ax.set_title(f'{display_name}\nDys Subgroup', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dys_scatter_audio_text_fusion_single.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Dys scatter plot saved to: {output_dir / 'dys_scatter_audio_text_fusion_single.png'}")
    
    # Reset font sizes to defaults
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    
    # Create summary table
    if all_predictions:
        summary_data = []
        for method_type, data in all_predictions.items():
            display_name = {'audio_only': 'Audio-Only', 'text_only': 'Best Text-Only', 'fusion': 'Best Fusion'}[method_type]
            summary_data.append({
                'Method': display_name,
                'Model': get_short_model_name(data['model'], config),
                'Model_Full': data['model'],
                'Method_Used': get_method_display_name(data['method'], config),
                'Method_Key': data['method'],
                'RMSE': data['rmse'],
                'R²': data['r2'],
                'N': data['n']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'dys_scatter_summary.csv', index=False)
        print(f"✓ Dys scatter summary saved to: {output_dir / 'dys_scatter_summary.csv'}")
        
        print("\n  Dys Scatter Plot Summary:")
        for _, row in summary_df.iterrows():
            print(f"    {row['Method']}:")
            print(f"      Model: {row['Model']} ({row['Model_Full']})")
            print(f"      Method: {row['Method_Used']} ({row['Method_Key']})")
            print(f"      RMSE: {row['RMSE']:.3f}, R²: {row['R²']:.3f}, n: {row['N']}")
    
    return all_predictions


# =======================================================================
#  BOOTSTRAP CONFIDENCE INTERVALS
# =======================================================================

def bootstrap_ci(data: np.ndarray, n_iterations: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence intervals for a metric."""
    if len(data) == 0 or np.all(np.isnan(data)):
        return np.nan, np.nan, np.nan
    
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        bootstrap_means.append(np.mean(data[indices]))
    
    bootstrap_means = np.array(bootstrap_means)
    
    if np.all(bootstrap_means == bootstrap_means[0]):
        mean = bootstrap_means[0]
        lower = mean
        upper = mean
    else:
        mean = np.mean(bootstrap_means)
        lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    if not np.isfinite(lower):
        lower = mean - 0.01
    if not np.isfinite(upper):
        upper = mean + 0.01
    
    return mean, lower, upper


def compute_confidence_intervals(df: pd.DataFrame, metric: str, group_cols: List[str], 
                                  n_iterations: int = 1000, ci: float = 0.95) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for grouped data."""
    results = []
    
    for name, group in df.groupby(group_cols):
        values = group[metric].values
        mean, lower, upper = bootstrap_ci(values, n_iterations, ci)
        
        if isinstance(name, tuple):
            row = {col: val for col, val in zip(group_cols, name)}
        else:
            row = {group_cols[0]: name}
        
        row[f'{metric}_mean'] = mean
        row[f'{metric}_lower_ci'] = lower
        row[f'{metric}_upper_ci'] = upper
        row[f'{metric}_std'] = np.nanstd(values) if len(values) > 0 else np.nan
        row['n_samples'] = len(values)
        
        if not np.isnan(mean) and not np.isnan(lower) and not np.isnan(upper):
            results.append(row)
    
    return pd.DataFrame(results)


# =======================================================================
#  BOOTSTRAP SUMMARY FUNCTION
# =======================================================================

def print_bootstrap_summary(ci_df: pd.DataFrame, metric: str, config: ExperimentConfig, 
                            task_type: str = 'classification'):
    """Print a summary of bootstrap confidence intervals."""
    if ci_df.empty:
        print(f"\n  No bootstrap results for {task_type}")
        return
    
    base_metric = metric.replace('_mean', '')
    metric_label = config.metric_labels.get(base_metric, base_metric.upper())
    
    print(f"\n  {'='*50}")
    print(f"  BOOTSTRAP SUMMARY: {task_type.upper()} - {metric_label}")
    print(f"  {'='*50}")
    print(f"  Bootstrap iterations: {ci_df['n_samples'].max() if not ci_df.empty else 'N/A'}")
    print(f"  Total model-method combinations: {len(ci_df)}")
    
    # Best performing combination
    if base_metric in ['rmse']:
        best_idx = ci_df[f'{base_metric}_mean'].idxmin()
    else:
        best_idx = ci_df[f'{base_metric}_mean'].idxmax()
    
    best_row = ci_df.loc[best_idx]
    print(f"\n  Best performing combination:")
    print(f"    Model: {best_row['Model']}")
    print(f"    Method: {best_row['Method_Label']}")
    print(f"    {metric_label}: {best_row[f'{base_metric}_mean']:.4f} (95% CI: {best_row[f'{base_metric}_lower_ci']:.4f} - {best_row[f'{base_metric}_upper_ci']:.4f})")
    
    # Best model across all methods
    model_means = ci_df.groupby('Model')[f'{base_metric}_mean'].mean()
    if base_metric in ['rmse']:
        best_model = model_means.idxmin()
        best_model_score = model_means.min()
    else:
        best_model = model_means.idxmax()
        best_model_score = model_means.max()
    print(f"\n  Best model (averaged across methods):")
    print(f"    Model: {best_model}")
    print(f"    Avg {metric_label}: {best_model_score:.4f}")
    
    # Best method across all models
    method_means = ci_df.groupby('Method_Label')[f'{base_metric}_mean'].mean()
    if base_metric in ['rmse']:
        best_method = method_means.idxmin()
        best_method_score = method_means.min()
    else:
        best_method = method_means.idxmax()
        best_method_score = method_means.max()
    print(f"\n  Best method (averaged across models):")
    print(f"    Method: {best_method}")
    print(f"    Avg {metric_label}: {best_method_score:.4f}")
    
    # CI width summary
    ci_widths = ci_df[f'{base_metric}_upper_ci'] - ci_df[f'{base_metric}_lower_ci']
    print(f"\n  Confidence Interval Widths:")
    print(f"    Min: {ci_widths.min():.4f}")
    print(f"    Max: {ci_widths.max():.4f}")
    print(f"    Mean: {ci_widths.mean():.4f}")
    print(f"    Std: {ci_widths.std():.4f}")


# =======================================================================
#  ABLATION ANALYSIS - USING ALL PATIENTS DATA (NOT SUBGROUPS)
# =======================================================================

def perform_ablation_analysis_all_patients(df, config, task_type='classification', verbose=True):
    if task_type == 'classification':
        metric = config.ranking_metric_classification
        lower_is_better = False
        metric_name = 'macro_f1'
    else:
        metric = config.ranking_metric_regression
        lower_is_better = True
        metric_name = 'r2'

    task_df = df[df['Task'] == task_type]
    if task_df.empty:
        return {}

    # ---- Ensemble labels defined HERE, unconditionally ----
    ensemble_labels = {
        'average ensemble',
        'voting ensemble',
        'stacking ensemble',
        'weighted ensemble',
        'confidence selection',
        'best method',
    }
    is_ensemble = task_df['Method_Label'].str.lower().str.strip().isin(ensemble_labels)
    base_df = task_df.copy()      # keep everything for ranking

    if verbose:
        print(f"\n  Ensemble labels recognised: {sorted(ensemble_labels)}")
        ens_in = task_df[is_ensemble]['Method_Label'].unique().tolist()
        print(f"  Ensemble rows present (kept for ranking): {sorted(ens_in)}")
    # ---- end ----

    # Filter to methods that have the overall metric
    base_df = base_df[base_df[metric_name].notna()].copy()

    if base_df.empty:
        if verbose:
            print(f"  Warning: No methods with {metric_name} for {task_type}")
        return {}

    all_methods = base_df['Method_Label'].unique()

    if verbose:
        print(f"\n  Base methods with {metric_name} for {task_type}: {len(all_methods)}")
        for m in sorted(all_methods):
            print(f"    ✓ {m}")
        # List excluded ensemble methods for clarity
        ensemble_methods = task_df[is_ensemble]['Method_Label'].unique()
        if len(ensemble_methods) > 0:
            print(f"\n  Excluded ensemble methods (not in ablation):")
            for m in sorted(ensemble_methods):
                print(f"    ✗ {m} (ensemble method)")

    # Find best method using the ranking metric
    method_performance = base_df.groupby('Method_Label')[metric_name].mean()
    if lower_is_better:
        best_method = method_performance.idxmin()
        best_score = method_performance.min()
    else:
        best_method = method_performance.idxmax()
        best_score = method_performance.max()

    if verbose:
        print(f"\n  Best base method: {best_method} ({best_score:.4f})")
    
    # Continue with ablation analysis on base methods only...
    ablation_results = []
    skipped_methods = []
    
    for method in all_methods:
        if method == best_method:
            continue
        
        method_df = base_df[base_df['Method_Label'] == method]
        best_df = base_df[base_df['Method_Label'] == best_method]
        
        # Get paired data
        method_values = []
        best_values = []
        paired_models = []
        
        for model in method_df['Model'].unique():
            m_val = method_df[method_df['Model'] == model][metric_name].values
            b_val = best_df[best_df['Model'] == model][metric_name].values
            
            if len(m_val) > 0 and len(b_val) > 0 and not np.isnan(m_val[0]) and not np.isnan(b_val[0]):
                method_values.append(m_val[0])
                best_values.append(b_val[0])
                paired_models.append(model)
        
        if len(method_values) < 2:
            skipped_methods.append((method, f"only {len(method_values)} paired samples"))
            continue
        
        method_values = np.array(method_values)
        best_values = np.array(best_values)
        
        diff = best_values - method_values if not lower_is_better else method_values - best_values
        
        try:
            t_stat, p_value_ttest = ttest_rel(best_values, method_values)
        except:
            t_stat, p_value_ttest = np.nan, np.nan
        
        try:
            w_stat, p_value_wilcoxon = wilcoxon(best_values, method_values)
        except:
            w_stat, p_value_wilcoxon = np.nan, np.nan
        
        pooled_std = np.sqrt((np.std(best_values, ddof=1)**2 + np.std(method_values, ddof=1)**2) / 2)
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
            'significant': p_value_ttest < 0.05 if not np.isnan(p_value_ttest) else False
        })
    
    if verbose and skipped_methods:
        print(f"\n  Skipped methods (insufficient paired data):")
        for method, reason in skipped_methods:
            print(f"    ✗ {method}: {reason}")
    
    ablation_df = pd.DataFrame(ablation_results)
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values('Difference', ascending=False)
        if verbose:
            print(f"\n  Included in ablation analysis: {len(ablation_df)} methods")
    
    return {
        'best_method': best_method,
        'best_score': best_score,
        'ablation_results': ablation_df,
        'metric': metric,
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'skipped_methods': skipped_methods
    }


# =======================================================================
#  ROBUSTNESS SUMMARY FUNCTION  (CI + Ablation)
# =======================================================================

def plot_robustness_summary_all_patients(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                                        task_type: str = 'classification', n_iterations: int = 1000):
    """
    Robustness summary for a task:
      1) Cross-model bootstrap confidence intervals per method
      2) Ablation analysis (all patients) + plot
    """
    if df.empty:
        print(f"Warning: Empty dataframe for task {task_type}")
        return None, None

    task_df = df[df['Task'] == task_type]
    if task_df.empty:
        print(f"Warning: No data for task {task_type}")
        return None, None

    ci_df = None
    ablation_data = None

    if task_type == 'classification':
        metric = config.ranking_metric_classification
    else:
        metric = config.ranking_metric_regression

    # ------------------------------------------------------------------
    # 1) Confidence intervals — bootstrap ACROSS models, per method
    # ------------------------------------------------------------------
    try:
        print(f"\n[CI] Computing cross-model CIs for {task_type}...")
        ci_rows = []
        for method_label, g in task_df.groupby('Method_Label'):
            vals = g[metric].dropna().values
            if len(vals) < 2:
                print(f"  [CI] Skipping '{method_label}': only {len(vals)} model(s)")
                continue
            mean, lower, upper = bootstrap_ci(vals, n_iterations=n_iterations, ci=0.95)
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
            ci_df.to_csv(output_dir / f'confidence_intervals_{task_type}.csv', index=False)
            print(f"✓ Confidence intervals data saved to: "
                  f"{output_dir / f'confidence_intervals_{task_type}.csv'}")

            print(f"\n  Cross-model bootstrap CIs ({metric}, {n_iterations} iters):")
            for _, row in ci_df.sort_values(f'{metric}_mean', ascending=False).iterrows():
                print(f"    {row['Method_Label']:32s} "
                      f"{row[f'{metric}_mean']:.3f} "
                      f"[{row[f'{metric}_lower_ci']:.3f}, {row[f'{metric}_upper_ci']:.3f}] "
                      f"n_models={row['n_models']}")
        else:
            print(f"⚠ No method had ≥2 models with non-NaN {metric} values")
    except Exception as e:
        import traceback
        print(f"⚠ Error computing confidence intervals for {task_type}: {e}")
        traceback.print_exc()
        ci_df = None

    # ------------------------------------------------------------------
    # 2) Ablation analysis (ALL patients)
    # ------------------------------------------------------------------
    try:
        print(f"\n[ABLATION] Running ablation for {task_type}...")
        ablation_data = perform_ablation_analysis_all_patients(
            task_df, config, task_type, verbose=True,
        )

        if ablation_data and not ablation_data.get('ablation_results', pd.DataFrame()).empty:
            plot_ablation_results_all_patients(ablation_data, output_dir, config, task_type)

            ablation_df = ablation_data['ablation_results']
            print(f"\n  Ablation Summary for {task_type.upper()} (ALL Patients):")
            print(f"    Reference method: {ablation_data['best_method']} "
                  f"({ablation_data['best_score']:.3f})")
            if not ablation_df.empty:
                top = ablation_df.iloc[0]
                print(f"    Most impactful removal: {top['Removed_Component']} "
                      f"(Δ = {top['Difference']:.3f}, p = {top['p_value_ttest']:.3f})")
        else:
            print(f"⚠ No ablation results could be computed for {task_type}")
    except Exception as e:
        import traceback
        print(f"⚠ Error computing ablation analysis for {task_type}: {e}")
        traceback.print_exc()
        ablation_data = None

    return ci_df, ablation_data


# =======================================================================
#  ABLATION PLOT
# =======================================================================

def plot_ablation_results_all_patients(ablation_data: Dict, output_dir: Path, config: ExperimentConfig,
                                       task_type: str = 'classification'):
    """
    Plot ablation study results.

    Contract with the caller (perform_ablation_analysis_all_patients):
      - ablation_data['best_method']    : reference method's DISPLAY label
                                          (may be an ensemble, e.g. "Voting Ensemble")
      - ablation_data['best_score']     : reference's mean score
      - ablation_data['metric_name']    : 'macro_f1' or 'r2'
      - ablation_data['lower_is_better']: bool
      - ablation_data['ablation_results']: DataFrame with columns
            Removed_Component, Removed_Score, Difference,
            p_value_ttest, significant, effect_size, n_pairs

    Layout notes:
      - sharey=True: panel 2 inherits panel 1's y-axis.
      - The y-tick system is bypassed entirely: labels are drawn with
        ax1.text(...) using get_yaxis_transform(). This avoids the
        sharey+invert_yaxis label-loss bug on all Matplotlib versions.
      - subplots_adjust is the sole spacing mechanism (no tight_layout,
        no constrained_layout) so wspace takes effect exactly as written.
      - Reference is named only in the panel-1 title; no annotation box.
    """
    ablation_df = ablation_data.get('ablation_results')
    if ablation_df is None or ablation_df.empty:
        print("Warning: No ablation data to plot")
        return

    metric_name  = ablation_data['metric_name']
    metric_label = config.metric_labels.get(metric_name, metric_name.upper())
    best_method  = ablation_data['best_method']
    best_score   = ablation_data['best_score']
    lower_is_better = ablation_data['lower_is_better']

    # ------------------------------------------------------------------
    # 0. TRACE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"ABLATION PLOT TRACE ({task_type})")
    print("=" * 70)
    print(f"  Reference method : {best_method}")
    print(f"  Reference score  : {best_score:.3f}  ({metric_label}, "
          f"{'lower is better' if lower_is_better else 'higher is better'})")
    print(f"  Rows in ablation_results: {len(ablation_df)}")
    print(f"  Components present:")
    for comp in ablation_df['Removed_Component'].tolist():
        print(f"    - {comp}")

    # ------------------------------------------------------------------
    # 1. Filter ensembles + reference from the bars
    # ------------------------------------------------------------------
    ENSEMBLE_LABELS = {
        'average ensemble', 'voting ensemble', 'stacking ensemble',
        'weighted ensemble', 'confidence selection', 'best method',
        'meta-fusion', 'meta fusion',
    }

    def _is_ensemble(label: str) -> bool:
        if not isinstance(label, str):
            return False
        ll = label.lower().strip()
        if ll in ENSEMBLE_LABELS:
            return True
        if 'ensemble' in ll:
            return True
        if 'meta-fusion' in ll or ll == 'meta fusion':
            return True
        return False

    is_ens = ablation_df['Removed_Component'].apply(_is_ensemble)
    is_ref = ablation_df['Removed_Component'].str.lower() == str(best_method).lower()

    base_only = ablation_df[~is_ens & ~is_ref].copy()

    if base_only.empty:
        print("  ⚠ No base methods left after ensemble filter — "
              "falling back to all non-reference rows")
        plot_df = ablation_df[~is_ref].copy()
    else:
        plot_df = base_only

    excluded = ablation_df[is_ens | is_ref]['Removed_Component'].tolist()

    print(f"\n  Excluded from bars ({len(excluded)}): {excluded}")
    print(f"  Bars to plot       ({len(plot_df)}): "
          f"{plot_df['Removed_Component'].tolist()}")

    if plot_df.empty:
        print("Warning: Nothing left to plot after removing ensembles and reference")
        return

    # ------------------------------------------------------------------
    # 2. Sort, drop NaNs, prepare y-positions
    # ------------------------------------------------------------------
    plot_df = plot_df.sort_values('Difference', ascending=False).reset_index(drop=True)
    plot_df = plot_df.dropna(subset=['Difference', 'p_value_ttest']).reset_index(drop=True)
    if plot_df.empty:
        print("Warning: No valid (non-NaN) ablation rows to plot")
        return

    n_bars = len(plot_df)
    y_positions = np.arange(n_bars) * 0.5

    # ------------------------------------------------------------------
    # 3. Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(20, max(6.75, n_bars * 0.55)),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.22,     # room for the text labels drawn on panel 1
        right=0.95,
        top=0.88,
        bottom=0.12,
        wspace=0.55,
    )

    # =============================================================
    # Panel 1 — Performance Drop
    # =============================================================
    ax1 = axes[0]
    colors = ['#2ECC71' if sig else '#E74C3C' for sig in plot_df['significant']]

    bars = ax1.barh(
        y_positions, plot_df['Difference'].values,
        color=colors, alpha=0.85, height=0.25,
        edgecolor='black', linewidth=0.6,
    )

    for bar, val in zip(bars, plot_df['Difference'].values):
        if val >= 0:
            ax1.text(bar.get_width() + 0.003,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:+.3f}', va='center', ha='left',
                     fontsize=12, fontweight='bold')
        else:
            ax1.text(bar.get_width() - 0.003,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:+.3f}', va='center', ha='right',
                     fontsize=12, fontweight='bold')

    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=1.5)

    ax1.set_xlabel(f'Performance Drop (Δ{metric_label}) vs Reference', fontsize=12)
    ax1.set_ylabel('')
    ax1.set_title(
        f'Ablation vs Reference ({task_type.title()})\n'
        f'Reference: {best_method} = {best_score:.3f}',
        fontsize=14, fontweight='bold', pad=10,
    )
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.tick_params(axis='x', labelsize=11)
    ax1.invert_yaxis()          # invert FIRST

    # ---- Draw category labels with ax1.text (bypasses the tick system) ----
    for y, name in zip(y_positions, plot_df['Removed_Component'].values):
        ax1.text(-0.02, y, name,
                 transform=ax1.get_yaxis_transform(),
                 ha='right', va='center', fontsize=11)
    ax1.set_yticks([])          # remove default ticks entirely

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', alpha=0.85, edgecolor='black',
              linewidth=0.6, label='Significant (p < 0.05)'),
        Patch(facecolor='#E74C3C', alpha=0.85, edgecolor='black',
              linewidth=0.6, label='Not Significant'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right',
               fontsize=10, framealpha=0.95)

    # =============================================================
    # Panel 2 — Statistical Significance
    #   sharey=True + set_yticks([]) → no duplicate labels, no second invert
    # =============================================================
    ax2 = axes[1]

    p_values = plot_df['p_value_ttest'].fillna(1.0).values
    log_p = -np.log10(p_values + 1e-10)
    colors2 = ['#2ECC71' if sig else '#E74C3C' for sig in plot_df['significant']]

    bars2 = ax2.barh(
        y_positions, log_p,
        color=colors2, alpha=0.85, height=0.25,
        edgecolor='black', linewidth=0.6,
    )

    for bar, p_val in zip(bars2, p_values):
        label = f'{p_val:.3f}' if p_val >= 0.001 else f'{p_val:.1e}'
        ax2.text(bar.get_width() + 0.05,
                 bar.get_y() + bar.get_height() / 2,
                 label, va='center', ha='left',
                 fontsize=11, fontweight='bold')

    threshold = -np.log10(0.05)
    ax2.axvline(x=threshold, color='red', linestyle='--',
                alpha=0.8, linewidth=2, label='p = 0.05')

    # No tick labels on panel 2 — panel 1 owns them; leave sharey to align.
    ax2.set_yticks([])
    ax2.set_ylabel('')

    ax2.set_xlabel('−log10(p-value)  (paired t-test vs reference)', fontsize=13)
    ax2.set_title(
        f'Statistical Significance ({task_type.title()})',
        fontsize=14, fontweight='bold', pad=10,
    )
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.tick_params(axis='x', labelsize=11)
    # No ax2.invert_yaxis() — shared axis already inverted via ax1.

    # ------------------------------------------------------------------
    # 3b. Diagnostic — actual gap between panels
    # ------------------------------------------------------------------
    fig.canvas.draw()
    b1 = axes[0].get_position()
    b2 = axes[1].get_position()
    print(f"  [LAYOUT] Panel 1 x-extent: {b1.x0:.3f} → {b1.x1:.3f}")
    print(f"  [LAYOUT] Panel 2 x-extent: {b2.x0:.3f} → {b2.x1:.3f}")
    print(f"  [LAYOUT] Gap between panels (fig fraction): {b2.x0 - b1.x1:.3f}")
    print(f"  [LAYOUT] n_bars={n_bars}  plot_df rows={len(plot_df)}")

    # ------------------------------------------------------------------
    # 4. Save figure + CSVs
    # ------------------------------------------------------------------
    out_png = output_dir / f'ablation_analysis_all_patients_{task_type}.png'
    print(f"  [SAVE] Writing {out_png}  (parent exists: {output_dir.exists()})")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Ablation figure saved: {out_png}")

    summary_cols = ['Removed_Component', 'Removed_Score', 'Difference',
                    'p_value_ttest', 'significant', 'effect_size', 'n_pairs']
    summary_cols = [c for c in summary_cols if c in plot_df.columns]
    summary_df = plot_df[summary_cols].round(4)
    summary_df.to_csv(
        output_dir / f'ablation_summary_all_patients_{task_type}.csv',
        index=False,
    )
    print(f"✓ Ablation summary CSV saved: "
          f"{output_dir / f'ablation_summary_all_patients_{task_type}.csv'}")

    with open(output_dir / f'ablation_reference_{task_type}.txt', 'w') as f:
        f.write(f"Reference method: {best_method}\n")
        f.write(f"Reference {metric_label}: {best_score:.3f}\n")
        f.write(f"Direction: {'lower is better' if lower_is_better else 'higher is better'}\n")
        f.write(f"Total rows in ablation_results: {len(ablation_df)}\n")
        f.write(f"Rows excluded from bars (ensembles + reference): {excluded}\n")
        f.write(f"Base methods plotted ({len(plot_df)}):\n")
        for comp in plot_df['Removed_Component'].tolist():
            f.write(f"  - {comp}\n")
    print(f"✓ Ablation reference info saved: "
          f"{output_dir / f'ablation_reference_{task_type}.txt'}")

# =======================================================================
#  VISUALIZATION FUNCTIONS
# =======================================================================

def plot_confidence_intervals(ci_df: pd.DataFrame, metric: str, output_dir: Path, 
                              config: ExperimentConfig, task_type: str = 'classification',
                              title: str = None):
    """Plot confidence intervals for each model-method combination."""
    if ci_df.empty:
        print(f"Warning: No CI data for {metric}")
        return
    
    base_metric = metric.replace('_mean', '')
    
    ci_df = ci_df.dropna(subset=[f'{base_metric}_mean', f'{base_metric}_lower_ci', f'{base_metric}_upper_ci'])
    
    if ci_df.empty:
        print(f"Warning: No valid CI data after cleaning for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(ci_df) * 0.3)))
    
    ci_df = ci_df.sort_values(f'{base_metric}_mean', ascending=False)
    
    labels = []
    for _, row in ci_df.iterrows():
        short_model = get_short_model_name(row['Model'], config)
        labels.append(f"{short_model} - {row['Method_Label']}")
    
    y_pos = np.arange(len(ci_df))
    means = ci_df[f'{base_metric}_mean'].values
    lower = ci_df[f'{base_metric}_lower_ci'].values
    upper = ci_df[f'{base_metric}_upper_ci'].values
    
    lower = np.maximum(lower, means - 10)
    upper = np.minimum(upper, means + 10)
    
    xerr_lower = means - lower
    xerr_upper = upper - means
    xerr_lower = np.maximum(xerr_lower, 0.001)
    xerr_upper = np.maximum(xerr_upper, 0.001)
    
    methods = ci_df['Method_Label'].unique()
    color_map = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods)}
    colors = [color_map[row['Method_Label']] for _, row in ci_df.iterrows()]
    
    ax.errorbar(means, y_pos, xerr=[xerr_lower, xerr_upper], 
                fmt='o', color='black', capsize=3, elinewidth=1, alpha=0.3)
    ax.scatter(means, y_pos, c=colors, s=80, alpha=0.8, zorder=3)
    
    metric_label = config.metric_labels.get(base_metric, base_metric.upper())
    ax.set_xlabel(f'{metric_label} (95% CI)', fontsize=12)
    ax.set_ylabel('Model - Method', fontsize=12)
    
    title_text = title or f'{metric_label} with 95% Confidence Intervals ({task_type.title()})'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=method) for method, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'confidence_intervals_{base_metric}{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence intervals plot saved to: {output_dir / f'confidence_intervals_{base_metric}{task_suffix}.png'}")


def plot_method_comparison(df: pd.DataFrame, metric: str, output_dir: Path, 
                           config: ExperimentConfig, title: str = None, 
                           subgroup_prefix: str = None, task_type: str = 'classification'):
    """Create grouped bar chart comparing methods across models."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for {metric}")
        return
    
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    plot_df['Model_Short'] = plot_df['Model'].apply(lambda x: get_short_model_name(x, config))
    
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    
    pivot = plot_df.pivot_table(
        index='Model_Short',
        columns=method_col,
        values=metric,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"Warning: No data for method_comparison_{metric}")
        return
    
    lower_is_better = metric in ['rmse']
    
    if lower_is_better:
        best_vals = pivot.min(axis=1)
    else:
        best_vals = pivot.max(axis=1)
    pivot = pivot.loc[best_vals.sort_values(ascending=lower_is_better).index]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot.index) * 0.4)))
    
    pivot.plot(kind='barh', ax=ax, width=0.8, colormap='viridis')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
    
    metric_label = config.metric_labels.get(metric, metric.upper())
    title_text = title or f'{metric_label} by Model and Fusion Method ({task_type.title()})'
    if subgroup_prefix:
        group_name = 'Dys' if 'subgroup' in subgroup_prefix else 'Norm'
        title_text = f'{group_name} - {metric_label} by Model and Fusion Method ({task_type.title()})'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Model')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'method_comparison_{metric}{suffix}{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Method comparison saved to: {output_dir / f'method_comparison_{metric}{suffix}{task_suffix}.png'}")


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig,
                 subgroup_prefix: str = None, task_type: str = 'classification'):
    """Create heatmap comparing methods across models."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for heatmap_{metric}")
        return
    
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    plot_df['Model_Short'] = plot_df['Model'].apply(lambda x: get_short_model_name(x, config))
    
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    
    pivot = plot_df.pivot_table(
        index=method_col,
        columns='Model_Short',
        values=metric,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"Warning: Empty pivot for heatmap_{metric}")
        return
    
    pivot = pivot.dropna(axis=1, how='all')
    
    if pivot.empty:
        print(f"Warning: No data after dropping NaN for heatmap_{metric}")
        return
    
    lower_is_better = metric in ['rmse']
    cmap = 'RdYlGn_r'
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.6), 
                                   max(8, len(pivot.index) * 0.5)))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                cbar_kws={'label': config.metric_labels.get(metric, metric.upper())},
                linewidths=0.5, linecolor='white',
                ax=ax, annot_kws={'fontsize': 8})
    
    metric_label = config.metric_labels.get(metric, metric.upper())
    title_text = f'{metric_label} Comparison Across Models ({task_type.title()})'
    if subgroup_prefix:
        group_name = 'Dys' if 'subgroup' in subgroup_prefix else 'Norm'
        title_text = f'{group_name} - {metric_label} Comparison Across Models ({task_type.title()})'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Fusion Method')
    
    plt.tight_layout()
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'heatmap_{metric}{suffix}{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved to: {output_dir / f'heatmap_{metric}{suffix}{task_suffix}.png'}")


def plot_sen_spec_combined(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                           subgroup_prefix: str = None, task_type: str = 'classification'):
    """Create combined Sen/Spec plots - simplified with only 4 bars (2 Dys, 2 Typ)."""
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    plot_df['Model_Short'] = plot_df['Model'].apply(lambda x: get_short_model_name(x, config))
    
    sen_col_dys = 'subgroup_sensitivity'
    spec_col_dys = 'subgroup_specificity'
    sen_col_typ = 'non_subgroup_sensitivity'
    spec_col_typ = 'non_subgroup_specificity'
    
    if sen_col_dys not in plot_df.columns or spec_col_dys not in plot_df.columns:
        print(f"Warning: Subgroup Sen/Spec columns not found")
        return
    
    if sen_col_typ not in plot_df.columns or spec_col_typ not in plot_df.columns:
        print(f"Warning: Non-subgroup (Typ) Sen/Spec columns not found")
        return
    
    avg_data = plot_df.groupby('Model_Short').agg({
        sen_col_dys: 'mean',
        spec_col_dys: 'mean',
        sen_col_typ: 'mean',
        spec_col_typ: 'mean'
    }).reset_index()
    
    avg_data = avg_data.dropna(subset=[sen_col_dys, spec_col_dys, sen_col_typ, spec_col_typ])
    
    if avg_data.empty:
        print(f"Warning: No valid data for Sen/Spec after dropping NaN")
        return
    
    avg_data = avg_data.sort_values(sen_col_dys, ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(avg_data) * 0.6)))
    
    models = avg_data['Model_Short'].values
    x = np.arange(len(models))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, avg_data[sen_col_dys], width, 
                   label='Dys - Sensitivity', color='#E74C3C', alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, avg_data[spec_col_dys], width, 
                   label='Dys - Specificity', color='#E74C3C', alpha=0.5, hatch='//')
    bars3 = ax.bar(x + 0.5*width, avg_data[sen_col_typ], width, 
                   label='Typ - Sensitivity', color='#3498DB', alpha=0.9)
    bars4 = ax.bar(x + 1.5*width, avg_data[spec_col_typ], width, 
                   label='Typ - Specificity', color='#3498DB', alpha=0.5, hatch='//')
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    # Move title higher using pad
    ax.set_title(f'Sensitivity & Specificity: Dys vs Typ ({task_type.title()})', 
                fontsize=18, fontweight='bold', pad=40)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
    
    # Legend ABOVE the title (higher than bbox_to_anchor)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), fontsize=12, ncol=4,
              frameon=True, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'sen_spec_combined{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sen/Spec combined plot saved to: {output_dir / f'sen_spec_combined{task_suffix}.png'}")
    
    avg_data.to_csv(output_dir / f'sen_spec_data{task_suffix}.csv', index=False)
    print(f"✓ Sen/Spec data saved to: {output_dir / f'sen_spec_data{task_suffix}.csv'}")
    
    return avg_data


def plot_subgroup_sen_spec_comprehensive(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                                          task_type: str = 'classification'):
    """Create comprehensive Sen/Spec plots comparing Dys vs Typ - 4 bars with legend above title."""
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    plot_df['Model_Short'] = plot_df['Model'].apply(lambda x: get_short_model_name(x, config))
    
    subgroup_sen = 'subgroup_sensitivity'
    subgroup_spec = 'subgroup_specificity'
    non_subgroup_sen = 'non_subgroup_sensitivity'
    non_subgroup_spec = 'non_subgroup_specificity'
    
    if subgroup_sen not in plot_df.columns or subgroup_spec not in plot_df.columns:
        print(f"Warning: Subgroup Sen/Spec columns not found")
        return
    
    if non_subgroup_sen not in plot_df.columns or non_subgroup_spec not in plot_df.columns:
        print(f"Warning: Non-subgroup (Typ) Sen/Spec columns not found")
        return
    
    avg_data = plot_df.groupby('Model_Short').agg({
        subgroup_sen: 'mean',
        subgroup_spec: 'mean',
        non_subgroup_sen: 'mean',
        non_subgroup_spec: 'mean'
    }).reset_index()
    
    avg_data = avg_data.dropna(subset=[subgroup_sen, subgroup_spec, non_subgroup_sen, non_subgroup_spec])
    
    if avg_data.empty:
        print(f"Warning: No valid data for subgroup Sen/Spec after dropping NaN")
        return
    
    avg_data = avg_data.sort_values(subgroup_sen, ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(avg_data) * 0.6)))
    
    models = avg_data['Model_Short'].values
    x = np.arange(len(models))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, avg_data[subgroup_sen], width, 
                   label='Dys - Sensitivity', color='#E74C3C', alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, avg_data[subgroup_spec], width, 
                   label='Dys - Specificity', color='#E74C3C', alpha=0.5, hatch='//')
    bars3 = ax.bar(x + 0.5*width, avg_data[non_subgroup_sen], width, 
                   label='Typ - Sensitivity', color='#3498DB', alpha=0.9)
    bars4 = ax.bar(x + 1.5*width, avg_data[non_subgroup_spec], width, 
                   label='Typ - Specificity', color='#3498DB', alpha=0.5, hatch='//')
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    # Title with pad to make room for legend above
    ax.set_title(f'Sensitivity & Specificity: Dys vs Typ ({task_type.title()})', 
                fontsize=18, fontweight='bold', pad=60)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
    
    # Legend ABOVE the title
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.03), fontsize=12, ncol=4,
              frameon=True, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=14)
    
    #plt.tight_layout()
    task_suffix = f"_{task_type}"
    plt.subplots_adjust(top=0.78)
    plt.savefig(output_dir / f'subgroup_sen_spec_comprehensive_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comprehensive subgroup Sen/Spec plot saved to: {output_dir / f'subgroup_sen_spec_comprehensive_{task_type}.png'}")
    
    avg_data.to_csv(output_dir / f'subgroup_sen_spec_data_{task_type}.csv', index=False)
    print(f"✓ Subgroup Sen/Spec data saved to: {output_dir / f'subgroup_sen_spec_data_{task_type}.csv'}")
    
    return avg_data


def plot_subgroup_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig, 
                             task_type: str = 'classification'):
    """Create comprehensive subgroup comparison plots."""
    plot_subgroup_sen_spec_comprehensive(df, output_dir, config, task_type=task_type)
    
    for subgroup_prefix in ['subgroup', 'non_subgroup']:
        plot_sen_spec_combined(df, output_dir, config, 
                              subgroup_prefix=subgroup_prefix, 
                              task_type=task_type)
    
    plot_sen_spec_combined(df, output_dir, config, 
                          subgroup_prefix=None, 
                          task_type=task_type)
    
    if task_type == 'classification':
        metrics_to_compare = ['macro_f1', 'balanced_accuracy', 'roc_auc']
    else:
        metrics_to_compare = ['rmse', 'r2']
    
    for metric in metrics_to_compare:
        subgroup_metric = f'subgroup_{metric}'
        non_subgroup_metric = f'non_subgroup_{metric}'
        
        if subgroup_metric not in df.columns or non_subgroup_metric not in df.columns:
            print(f"  Warning: Subgroup metrics not found for {metric}")
            continue
        
        plot_data = df[df['Task'] == task_type].copy()
        plot_data['Model_Short'] = plot_data['Model'].apply(lambda x: get_short_model_name(x, config))
        
        plot_data = plot_data.groupby('Model_Short').agg({
            subgroup_metric: 'mean',
            non_subgroup_metric: 'mean'
        }).reset_index()
        
        if plot_data.empty:
            print(f"  Warning: No data for subgroup comparison {metric}")
            continue
        
        plot_data['diff'] = plot_data[subgroup_metric] - plot_data[non_subgroup_metric]
        
        if metric in ['rmse']:
            plot_data = plot_data.sort_values(subgroup_metric, ascending=True)
        else:
            plot_data = plot_data.sort_values(subgroup_metric, ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(plot_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, plot_data[subgroup_metric], width, 
                      label='Dys (Subgroup)', color='#E74C3C', alpha=0.8)
        bars2 = ax.bar(x + width/2, plot_data[non_subgroup_metric], width, 
                      label='Norm (Non-Subgroup)', color='#3498DB', alpha=0.8)
        
        for bar in bars1:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        metric_label = config.metric_labels.get(metric, metric.upper())
        ax.set_title(f'{metric_label}: Dys vs Norm Comparison ({task_type.title()})', 
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
        task_suffix = f"_{task_type}"
        plt.savefig(output_dir / f'subgroup_comparison_{metric}{task_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Subgroup comparison saved to: {output_dir / f'subgroup_comparison_{metric}{task_suffix}.png'}")


def create_summary_table(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig, prefix: str = ''):
    """Create summary table with mean and std for each model-method."""
    tasks = df['Task'].unique()
    
    for task in tasks:
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue
        
        if task == 'classification':
            metrics = [m for m in config.classification_metrics if m in task_df.columns]
            subgroup_metrics = [m for m in config.classification_subgroup_metrics if m in task_df.columns]
        else:
            metrics = [m for m in config.regression_metrics if m in task_df.columns]
            subgroup_metrics = [m for m in config.regression_subgroup_metrics if m in task_df.columns]
        
        all_metrics = metrics + subgroup_metrics
        
        agg_dict = {}
        for metric in all_metrics:
            if metric in task_df.columns:
                agg_dict[metric] = ['mean', 'std', 'count']
        
        if not agg_dict:
            print(f"  Warning: No metrics found for {task}")
            continue
        
        task_df['Model_Short'] = task_df['Model'].apply(lambda x: get_short_model_name(x, config))
        
        group_cols = ['Model_Short']
        if 'Method_Label' in task_df.columns:
            group_cols.append('Method_Label')
        elif 'Method' in task_df.columns:
            group_cols.append('Method')
        
        try:
            if len(group_cols) > 1:
                summary = task_df.groupby(group_cols).agg(agg_dict)
            else:
                summary = task_df.groupby('Model_Short').agg(agg_dict)
        except KeyError as e:
            print(f"  Warning: Groupby failed for {task}: {e}")
            summary = task_df.groupby('Model_Short').agg(agg_dict)
        
        summary = summary.round(4)
        
        filename = f'{prefix}summary_table_{task}.csv' if prefix else f'summary_table_{task}.csv'
        summary.to_csv(output_dir / filename)
        print(f"✓ Summary table saved to: {output_dir / filename}")
        
        flat_summary = summary.copy()
        flat_summary.columns = [f'{col[0]}_{col[1]}' for col in flat_summary.columns]
        flat_filename = f'{prefix}summary_table_flat_{task}.csv' if prefix else f'summary_table_flat_{task}.csv'
        flat_summary.reset_index().to_csv(output_dir / flat_filename, index=False)



# =======================================================================
#  MAIN FUNCTION
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment Results Aggregator - Supports Classification & Regression with Robustness Analysis'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=str, default='./results_summary',
                        help='Output directory for summary and figures')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'all'], 
                        default='all', help='Task type to aggregate')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to include')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific fusion methods to include')
    parser.add_argument('--ignore-methods', nargs='+', default=None,
                        help='Fusion methods to ignore/exclude (e.g., mlp cca)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to include in figures')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Select top K models based on ranking metric')
    parser.add_argument('--dys-ids', type=str, default=None,
                        help='Text file with Dys speaker IDs (one per line)')
    parser.add_argument('--bootstrap-iterations', type=int, default=1000,
                        help='Number of bootstrap iterations for confidence intervals (default: 1000)')
    parser.add_argument('--no-robustness', action='store_true',
                        help='Skip robustness analysis (CI and ablation)')
    parser.add_argument('--no-scatter', action='store_true',
                        help='Skip scatter plots for Dys subgroup')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--subgroup', action='store_true',
                        help='Generate subgroup (Dys/Norm) analysis')

    # In argparse, add option to filter by method group
    parser.add_argument('--method-group', type=str, 
                        choices=['all', 'base', 'advanced', 'sota', 'meta'],
                        default='all',
                        help='Filter methods by group: base (4), advanced (7), sota (3), meta, or all')

    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ExperimentConfig()
    config.bootstrap_iterations = args.bootstrap_iterations
    
    # Load Dys speaker IDs if provided
    dys_ids = []
    if args.dys_ids:
        dys_ids = load_dys_speaker_ids(Path(args.dys_ids))
        if not dys_ids:
            print("Warning: No Dys speaker IDs loaded. Scatter plots will use all data.")
    else:
        print("\nNo Dys speaker ID file provided. Scatter plots will use all data.")
        print("To filter for Dys subgroup, provide --dys-ids with a file containing speaker IDs.")
    
    base_dir = Path(args.input_dir)
    experiments = discover_experiments(base_dir, args.task, config)
    
    if not experiments:
        print("\n" + "="*60)
        print("ERROR: No experiments found!")
        print("="*60)
        print(f"Base directory: {base_dir}")
        return
    
    if args.models:
        print(f"\n{'='*60}")
        print(f"FILTERING MODELS")
        print(f"{'='*60}")
        print(f"Including only models: {args.models}")
        
        filtered_experiments = {}
        for model_name, model_data in experiments.items():
            include_model = False
            for allowed_model in args.models:
                if allowed_model in model_name or model_name in allowed_model:
                    include_model = True
                    break
            
            if include_model:
                filtered_experiments[model_name] = model_data
                print(f"  ✓ Including: {model_name}")
            else:
                print(f"  ✗ Excluding: {model_name}")
        
        experiments = filtered_experiments
        
        if not experiments:
            print("\n" + "="*60)
            print("ERROR: No experiments match the specified models!")
            print("="*60)
            return
        
        print(f"\n{len(experiments)} models selected")
    
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    df = aggregate_experiment_results(experiments, config)
    print(f"Aggregated {len(df)} method results")
    
    if args.ignore_methods:
        ignore_set = set(args.ignore_methods)
        df = df[~df['Method'].isin(ignore_set) & ~df['Method_Label'].isin(ignore_set)]
        print(f"  Removed methods: {args.ignore_methods}")
    
    if args.top_k and args.top_k > 0:
        print(f"\n{'='*60}")
        print(f"SELECTING TOP {args.top_k} MODELS")
        print(f"{'='*60}")
        
        df = select_top_k_models(df, args.top_k, args.task if args.task != 'all' else None, config)
        print(f"\nFiltered to {len(df['Model'].unique())} models after top-k selection")
    
    if args.task != 'all':
        df = df[df['Task'] == args.task]
    
    if args.models:
        df = df[df['Model'].isin(args.models)]
    
    if args.methods:
        df = df[df['Method'].isin(args.methods)]
    
    if df.empty:
        print("\n" + "="*60)
        print("ERROR: No data found after filtering!")
        print("="*60)
        return
    
    print("\nTasks found:", df['Task'].unique().tolist())
    print("Models found:", df['Model'].unique().tolist())
    print("Model sizes:", df['Size'].unique().tolist())
    print("Model families:", df['Family'].unique().tolist())
    print("Methods found:", df['Method_Label'].unique().tolist())
    
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"GENERATING SUMMARY TABLES")
    print(f"{'='*60}")
    
    create_summary_table(df, output_dir, config)
    
    # ===== ROBUSTNESS ANALYSIS (ALL PATIENTS) =====
    if not args.no_robustness:
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS ANALYSIS (Bootstrap CI & Ablation - ALL Patients)")
        print(f"{'='*60}")
        print(f"Bootstrap iterations: {args.bootstrap_iterations}")
        
        tasks_to_process = df['Task'].unique()
        
        for task in tasks_to_process:
            print(f"\nProcessing {task} task...")
            plot_robustness_summary_all_patients(df, output_dir, config, task, args.bootstrap_iterations)
    
    # ===== SCATTER PLOTS FOR DYS SUBGROUP =====
    if not args.no_scatter and 'regression' in df['Task'].unique():
        print(f"\n{'='*60}")
        print(f"GENERATING DYS SCATTER PLOTS")
        print(f"{'='*60}")
        plot_dys_scatter_audio_text_fusion_single(df, experiments, output_dir, config, 'regression', dys_ids)
    
    # ===== PLOTS =====
        # ===== PLOTS =====
    if not args.no_plots:
        print(f"\n{'='*60}")
        print(f"GENERATING FIGURES")
        print(f"{'='*60}")
        
        tasks_to_process = df['Task'].unique()
        
        for task in tasks_to_process:
            print(f"\nProcessing {task} task...")
            
            if task == 'classification':
                default_metrics = ['macro_f1', 'sensitivity', 'specificity', 'balanced_accuracy', 'roc_auc']
                all_metrics = config.classification_metrics + config.classification_subgroup_metrics
            else:
                default_metrics = ['rmse', 'r2']
                all_metrics = config.regression_metrics + config.regression_subgroup_metrics
            
            if args.metrics:
                metrics_to_plot = [m for m in args.metrics if m in all_metrics]
            else:
                metrics_to_plot = [m for m in default_metrics if m in df.columns and df[df['Task'] == task][m].notna().any()]
            
            for metric in metrics_to_plot:
                if metric in df.columns and df[df['Task'] == task][metric].notna().any():
                    plot_heatmap(df, metric, output_dir, config, task_type=task)
                    plot_method_comparison(df, metric, output_dir, config, task_type=task)
            
            if args.subgroup:
                plot_subgroup_comparison(df, output_dir, config, task_type=task)

        # ===== META-FUSION ANALYSIS =====
        # Check if we have meta-fusion results
        has_meta = any(df['Method'].str.startswith('ensemble_') | (df['Method'] == 'meta_fusion'))
        if has_meta:
            print(f"\n{'='*60}")
            print(f"GENERATING META-FUSION FIGURES")
            print(f"{'='*60}")
            
            for task in tasks_to_process:
                plot_meta_fusion_comparison(df, output_dir, config, task_type=task)
    
    # ===== SUMMARY STATISTICS =====
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    for task in df['Task'].unique(): 
        task_df = df[df['Task'] == task]
        print(f"\n{task.upper()}:")
        
        if task == 'classification':
            ranking_metric = config.ranking_metric_classification
        else:
            ranking_metric = config.ranking_metric_regression
        
        if ranking_metric in task_df.columns:
            metric_label = config.metric_labels.get(ranking_metric, ranking_metric.upper())
            
            print(f"\n  Best by {metric_label}:")
            best_by_method = task_df.groupby('Method_Label')[ranking_metric].mean()
            
            if ranking_metric in ['rmse']:
                best_by_method = best_by_method.sort_values(ascending=True)
            else:
                best_by_method = best_by_method.sort_values(ascending=False)
            
            for method, val in best_by_method.head(5).items():
                print(f"    {method}: {val:.4f}")
            
            print(f"\n  Best by Model:")
            best_by_model = task_df.groupby('Model')[ranking_metric].mean()
            if ranking_metric in ['rmse']:
                best_by_model = best_by_model.sort_values(ascending=True)
            else:
                best_by_model = best_by_model.sort_values(ascending=False)
            for model, val in best_by_model.head(5).items():
                short_name = get_short_model_name(model, config)
                print(f"    {short_name}: {val:.4f}")
            
            if args.subgroup:
                subgroup_metric = f'subgroup_{ranking_metric}'
                non_subgroup_metric = f'non_subgroup_{ranking_metric}'
                if subgroup_metric in task_df.columns and non_subgroup_metric in task_df.columns:
                    print(f"\n  Dys vs Norm Comparison ({metric_label}):")
                    dys_avg = task_df[subgroup_metric].mean()
                    norm_avg = task_df[non_subgroup_metric].mean()
                    print(f"    Dys: {dys_avg:.4f}")
                    print(f"    Norm: {norm_avg:.4f}")
                    print(f"    Difference: {dys_avg - norm_avg:.4f}")
                
                if task == 'classification':
                    for sen_spec in ['sensitivity', 'specificity']:
                        subgroup_metric = f'subgroup_{sen_spec}'
                        non_subgroup_metric = f'non_subgroup_{sen_spec}'
                        if subgroup_metric in task_df.columns and non_subgroup_metric in task_df.columns:
                            dys_avg = task_df[subgroup_metric].mean()
                            norm_avg = task_df[non_subgroup_metric].mean()
                            metric_label_sen = config.metric_labels.get(sen_spec, sen_spec.title())
                            print(f"    Dys vs Norm {metric_label_sen}: Dys={dys_avg:.4f}, Norm={norm_avg:.4f}, Diff={dys_avg - norm_avg:.4f}")
    
    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*60}")
    print(f"\nAdditional outputs:")
    print(f"  - Confidence intervals: confidence_intervals_*.csv and plots")
    print(f"  - Bootstrap summary printed above")
    print(f"  - Ablation analysis (ALL patients): ablation_*_all_patients_*.csv and plots")
    print(f"  - Dys scatter plots: dys_scatter_*.png")
    print(f"  - Dys scatter summary: dys_scatter_summary.csv (includes which models were used)")
    print(f"  - Bootstrap iterations: {args.bootstrap_iterations}")


if __name__ == "__main__":
    main()

'''
python ~/asr_clinical/question_ensemble_fusion_aggregate_results.py --input-dir outputs-ensemble --output-dir outputs-ensemble-aggregate --subgroup --top-k 5 --bootstrap-iterations 10000 --ignore-methods mlp --verbose --dys-ids dysarthria-list.txt| tee outputs-ensemble-aggregate/log.txt
'''