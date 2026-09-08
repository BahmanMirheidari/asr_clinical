"""
Experiment Results Aggregator - Enhanced Version
Supports BOTH classification and regression experiments with subgroup analysis (Dys/Norm).

Key Metrics:
- Classification: F1-Macro, Sensitivity, Specificity, Balanced Accuracy, AUC-ROC
- Regression: RMSE, R²
- Subgroup analysis: Dys vs Norm for all metrics with Sen/Spec pairs
- Robustness: Confidence Intervals (95%), Bootstrap resampling
- Ablation: Component removal analysis with statistical significance
- Statistical tests: t-test, Wilcoxon signed-rank test

Folder Structure:
<main_dir>/
├── classification-fusion-<model_name>/           
│   ├── fusion_results/
│   │   ├── leakage_safe_5fold/
│   │   │   ├── audio_only_aggregate_metrics.json
│   │   │   ├── text_only_aggregate_metrics.json
│   │   │   ├── fuse-audio_text_aggregate_metrics.json
│   │   │   └── ...
│   │   └── meta_fusion/
│   │       ├── meta_fusion_metrics.json
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
from scipy.stats import ttest_rel, wilcoxon, ttest_ind
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
    
    # Standard fusion methods
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
        'dynamic': 'dynamic_aggregate_metrics.json'
    })
    
    # Display names for standard fusion methods
    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        'audio_only': 'Audio-Only',
        'text_only': 'Text-Only',
        'early': 'Early Fusion',
        'late': 'Late Fusion',
        'confidence': 'Confidence-Weighted',
        'interaction': 'Interaction Stacking',
        'moe': 'Mixture of Experts',
        'mlp': 'MLP Early Fusion',
        'stacking': 'Model-Based Stacking',
        'cca': 'CCA Fusion',
        'dynamic': 'Dynamic Fusion'
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
    })
    
    # Meta-fusion display names
    meta_fusion_display_names: Dict[str, str] = field(default_factory=lambda: {
        'average': 'Average Ensemble',
        'voting': 'Voting Ensemble',
        'stacking': 'Stacking Ensemble',
        'weighted': 'Weighted Ensemble',
        'confidence_selection': 'Confidence Selection',
        'best': 'Best Method'
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
        'non_subgroup_macro_f1': 'Norm - F1-Macro',
        'non_subgroup_sensitivity': 'Norm - Sensitivity',
        'non_subgroup_specificity': 'Norm - Specificity',
        'non_subgroup_balanced_accuracy': 'Norm - Balanced Accuracy',
        'non_subgroup_roc_auc': 'Norm - AUC-ROC',
        'subgroup_rmse': 'Dys - RMSE',
        'subgroup_r2': 'Dys - R²',
        'non_subgroup_rmse': 'Norm - RMSE',
        'non_subgroup_r2': 'Norm - R²'
    })
    
    # Ranking metrics
    ranking_metric_classification: str = 'macro_f1'
    ranking_metric_regression: str = 'r2'
    
    # Bootstrap iterations
    bootstrap_iterations: int = 1000


# =======================================================================
#  HELPER FUNCTIONS
# =======================================================================

def get_method_display_name(method_key: str, config: ExperimentConfig) -> str:
    """Get display name for a method."""
    if method_key.startswith('fuse-'):
        if method_key in config.fusion_combination_display_names:
            return config.fusion_combination_display_names[method_key]
        else:
            parts = method_key.replace('fuse-', '').split('_')
            return ' + '.join([p.replace('_', ' ').title() for p in parts])
    else:
        return config.method_display_names.get(method_key, method_key.replace('_', ' ').title())


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


# =======================================================================
#  EXPERIMENT DISCOVERY
# =======================================================================

def discover_experiments(base_dir: Path, task_type: str = 'all', config: ExperimentConfig = None) -> Dict:
    """Discover experiments by scanning folders."""
    if config is None:
        config = ExperimentConfig()
    
    experiments = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n{'='*60}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*60}")
    print(f"Task type filter: {task_type}")
    
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
            json_files = list(leakage_dir.glob("*_aggregate_metrics.json"))
            json_files.extend(leakage_dir.glob("*_metrics.json"))
            json_files = list(set(json_files))
            
            print(f"    Found {len(json_files)} metric files in leakage_safe_5fold")
            
            for json_file in json_files:
                filename = json_file.name
                method_key = None
                
                for key, fname in config.fusion_methods.items():
                    if filename == fname:
                        method_key = key
                        break
                
                if method_key is None:
                    base_name = filename
                    for suffix in ['_aggregate_metrics.json', '_metrics.json', '.json']:
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
                    'task': task,
                    'model': model_name,
                    'size': size,
                    'family': family,
                    'metrics_file': json_file,
                    'dir': folder_path,
                    'source': 'leakage_safe_5fold'
                }
                found_fusion = True
                print(f"    ✓ Found: {method_key} -> {json_file.name}")
        else:
            print(f"    ✗ leakage_safe_5fold NOT found")
        
        if meta_fusion_dir.exists():
            meta_fusion_file = meta_fusion_dir / 'meta_fusion_metrics.json'
            
            if meta_fusion_file.exists():
                meta_metrics = load_metrics_file(meta_fusion_file)
                if meta_metrics:
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
        else:
            print(f"    ✗ meta_fusion NOT found")
        
        if not found_fusion:
            print(f"    ⚠ No fusion metrics found in leakage_safe_5fold")
    
    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    
    total_methods = 0
    classification_models = 0
    regression_models = 0
    
    for model_key, model_data in experiments.items():
        task = 'unknown'
        for method_key, method_data in model_data.items():
            if 'task' in method_data:
                task = method_data['task']
                break
        
        if task == 'classification':
            classification_models += 1
        else:
            regression_models += 1
        
        print(f"\nTask: {task}, Model: {model_key}")
        method_count = len([m for m in model_data.keys() if not m.startswith('ensemble_')])
        ensemble_count = len([m for m in model_data.keys() if m.startswith('ensemble_')])
        print(f"  {method_count} fusion methods")
        if ensemble_count > 0:
            print(f"  {ensemble_count} ensemble methods")
        total_methods += method_count + ensemble_count
    
    print(f"\nTotal: {len(experiments)} models ({classification_models} classification, {regression_models} regression)")
    print(f"Total method results: {total_methods}")
    
    return dict(experiments)


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
#  BOOTSTRAP CONFIDENCE INTERVALS
# =======================================================================

def bootstrap_ci(data: np.ndarray, n_iterations: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        data: Array of values
        n_iterations: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95)
    
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
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
    
    # Handle case where all bootstrap means are the same
    if np.all(bootstrap_means == bootstrap_means[0]):
        mean = bootstrap_means[0]
        lower = mean
        upper = mean
    else:
        mean = np.mean(bootstrap_means)
        lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    # Ensure lower and upper are finite
    if not np.isfinite(lower):
        lower = mean - 0.01
    if not np.isfinite(upper):
        upper = mean + 0.01
    
    return mean, lower, upper


def compute_confidence_intervals(df: pd.DataFrame, metric: str, group_cols: List[str], 
                                  n_iterations: int = 1000, ci: float = 0.95) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for grouped data.
    
    Args:
        df: DataFrame with results
        metric: Metric to compute CI for
        group_cols: Columns to group by
        n_iterations: Number of bootstrap iterations
        ci: Confidence interval level
    
    Returns:
        DataFrame with mean, lower_ci, upper_ci for each group
    """
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
        
        # Only add valid rows
        if not np.isnan(mean) and not np.isnan(lower) and not np.isnan(upper):
            results.append(row)
    
    return pd.DataFrame(results)


# =======================================================================
#  ABLATION ANALYSIS
# =======================================================================

def perform_ablation_analysis(df: pd.DataFrame, config: ExperimentConfig, 
                             task_type: str = 'classification') -> Dict:
    """
    Perform ablation analysis: compare full model vs removing each component.
    
    Returns:
        Dictionary with ablation results for each component
    """
    if task_type == 'classification':
        metric = config.ranking_metric_classification
        lower_is_better = False
    else:
        metric = config.ranking_metric_regression
        lower_is_better = True
    
    task_df = df[df['Task'] == task_type]
    if task_df.empty:
        return {}
    
    all_methods = task_df['Method_Label'].unique()
    
    method_performance = task_df.groupby('Method_Label')[metric].mean()
    if lower_is_better:
        best_method = method_performance.idxmin()
        best_score = method_performance.min()
    else:
        best_method = method_performance.idxmax()
        best_score = method_performance.max()
    
    ablation_results = []
    
    for method in all_methods:
        if method == best_method:
            continue
        
        method_df = task_df[task_df['Method_Label'] == method]
        best_df = task_df[task_df['Method_Label'] == best_method]
        
        method_values = []
        best_values = []
        
        for model in method_df['Model'].unique():
            m_val = method_df[method_df['Model'] == model][metric].values
            b_val = best_df[best_df['Model'] == model][metric].values
            
            if len(m_val) > 0 and len(b_val) > 0 and not np.isnan(m_val[0]) and not np.isnan(b_val[0]):
                method_values.append(m_val[0])
                best_values.append(b_val[0])
        
        if len(method_values) < 2:
            continue
        
        method_values = np.array(method_values)
        best_values = np.array(best_values)
        
        diff = best_values - method_values if not lower_is_better else method_values - best_values
        
        t_stat, p_value_ttest = ttest_rel(best_values, method_values)
        
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
            'significant': p_value_ttest < 0.05
        })
    
    ablation_df = pd.DataFrame(ablation_results)
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values('Difference', ascending=False)
    
    return {
        'best_method': best_method,
        'best_score': best_score,
        'ablation_results': ablation_df,
        'metric': metric,
        'lower_is_better': lower_is_better
    }


# =======================================================================
#  ROBUSTNESS VISUALIZATIONS
# =======================================================================

def plot_confidence_intervals(ci_df: pd.DataFrame, metric: str, output_dir: Path, 
                              config: ExperimentConfig, task_type: str = 'classification',
                              title: str = None):
    """Plot confidence intervals for each model-method combination."""
    if ci_df.empty:
        print(f"Warning: No CI data for {metric}")
        return
    
    base_metric = metric.replace('_mean', '')
    
    # Clean data - remove rows with NaN or invalid values
    ci_df = ci_df.dropna(subset=[f'{base_metric}_mean', f'{base_metric}_lower_ci', f'{base_metric}_upper_ci'])
    
    if ci_df.empty:
        print(f"Warning: No valid CI data after cleaning for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(ci_df) * 0.3)))
    
    ci_df = ci_df.sort_values(f'{base_metric}_mean', ascending=False)
    
    labels = [f"{row['Model']} - {row['Method_Label']}" for _, row in ci_df.iterrows()]
    
    y_pos = np.arange(len(ci_df))
    means = ci_df[f'{base_metric}_mean'].values
    lower = ci_df[f'{base_metric}_lower_ci'].values
    upper = ci_df[f'{base_metric}_upper_ci'].values
    
    # Ensure lower <= upper and values are finite
    lower = np.maximum(lower, means - 10)  # Prevent extreme negative values
    upper = np.minimum(upper, means + 10)
    
    # Calculate xerr safely
    xerr_lower = means - lower
    xerr_upper = upper - means
    
    # Replace any negative or NaN values with small positive values
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
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=method) for method, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'confidence_intervals_{base_metric}{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence intervals saved to: {output_dir / f'confidence_intervals_{base_metric}{task_suffix}.png'}")

def plot_ablation_results(ablation_data: Dict, output_dir: Path, config: ExperimentConfig,
                         task_type: str = 'classification'):
    """Plot ablation study results."""
    ablation_df = ablation_data['ablation_results']
    if ablation_df.empty:
        print("Warning: No ablation data to plot")
        return
    
    metric = ablation_data['metric']
    metric_label = config.metric_labels.get(metric, metric.upper())
    best_method = ablation_data['best_method']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(ablation_df) * 0.4)))
    
    ablation_df = ablation_df.sort_values('Difference', ascending=False)
    
    ax1 = axes[0]
    colors = ['#2ECC71' if row['significant'] else '#E74C3C' for _, row in ablation_df.iterrows()]
    
    bars = ax1.barh(ablation_df['Removed_Component'], ablation_df['Difference'], color=colors, alpha=0.7)
    
    for bar, val in zip(bars, ablation_df['Difference']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel(f'Performance Drop (Δ{metric_label})', fontsize=12)
    ax1.set_ylabel('Removed Component', fontsize=12)
    ax1.set_title(f'Ablation: Performance Drop When Removing Component\n(Compared to {best_method})', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='#E74C3C', alpha=0.7, label='Not Significant')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=9)
    
    ax2 = axes[1]
    
    p_values = ablation_df['p_value_ttest'].values
    significant = ablation_df['significant'].values
    
    log_p = -np.log10(p_values + 1e-10)
    
    colors2 = ['#2ECC71' if sig else '#E74C3C' for sig in significant]
    bars2 = ax2.barh(ablation_df['Removed_Component'], log_p, color=colors2, alpha=0.7)
    
    for bar, p_val in zip(bars2, p_values):
        label = f'{p_val:.4f}' if p_val >= 0.001 else f'{p_val:.2e}'
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=8)
    
    threshold = -np.log10(0.05)
    ax2.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    
    ax2.set_xlabel('-log10(p-value)', fontsize=12)
    ax2.set_ylabel('Removed Component', fontsize=12)
    ax2.set_title('Statistical Significance of Ablation', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'ablation_analysis_{task_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ablation analysis saved to: {output_dir / f'ablation_analysis_{task_type}.png'}")
    
    summary_cols = ['Removed_Component', 'Removed_Score', 'Difference', 
                    'p_value_ttest', 'significant', 'effect_size', 'n_pairs']
    summary_df = ablation_df[summary_cols].round(4)
    summary_df.to_csv(output_dir / f'ablation_summary_{task_type}.csv', index=False)
    print(f"✓ Ablation summary saved to: {output_dir / f'ablation_summary_{task_type}.csv'}")


def plot_robustness_summary(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                           task_type: str = 'classification', n_iterations: int = 1000):
    """Create comprehensive robustness summary with CIs and statistics."""
    if df.empty:
        print(f"Warning: Empty dataframe for task {task_type}")
        return
    
    task_df = df[df['Task'] == task_type]
    if task_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    if task_type == 'classification':
        metric = config.ranking_metric_classification
    else:
        metric = config.ranking_metric_regression
    
    try:
        ci_df = compute_confidence_intervals(
            task_df, metric, ['Model', 'Method_Label'], 
            n_iterations=n_iterations, ci=0.95
        )
        
        if not ci_df.empty:
            plot_confidence_intervals(ci_df, metric, output_dir, config, task_type)
            ci_df.to_csv(output_dir / f'confidence_intervals_{task_type}.csv', index=False)
            print(f"✓ Confidence intervals data saved to: {output_dir / f'confidence_intervals_{task_type}.csv'}")
        else:
            print(f"⚠ No confidence intervals could be computed for {task_type}")
    except Exception as e:
        print(f"⚠ Error computing confidence intervals for {task_type}: {e}")
    
    try:
        ablation_data = perform_ablation_analysis(task_df, config, task_type)
        
        if ablation_data and not ablation_data['ablation_results'].empty:
            ablation_df = ablation_data['ablation_results']
            ablation_df.to_csv(output_dir / f'ablation_detailed_{task_type}.csv', index=False)
            print(f"✓ Detailed ablation results saved to: {output_dir / f'ablation_detailed_{task_type}.csv'}")
            
            plot_ablation_results(ablation_data, output_dir, config, task_type)
            
            print(f"\n  Ablation Summary for {task_type.upper()}:")
            print(f"    Best Method: {ablation_data['best_method']} ({ablation_data['best_score']:.4f})")
            if not ablation_df.empty:
                print(f"    Most impactful removal: {ablation_df.iloc[0]['Removed_Component']} "
                      f"(Δ = {ablation_df.iloc[0]['Difference']:.4f}, p = {ablation_df.iloc[0]['p_value_ttest']:.4f})")
        else:
            print(f"⚠ No ablation results could be computed for {task_type}")
    except Exception as e:
        print(f"⚠ Error computing ablation analysis for {task_type}: {e}")
    
    return ci_df if 'ci_df' in locals() else None, ablation_data if 'ablation_data' in locals() else None


# =======================================================================
#  VISUALIZATION FUNCTIONS (ORIGINAL)
# =======================================================================

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
    
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    
    pivot = plot_df.pivot_table(
        index='Model',
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
    
    method_col = 'Method_Label' if 'Method_Label' in plot_df.columns else 'Method'
    
    pivot = plot_df.pivot_table(
        index=method_col,
        columns='Model',
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
    """Create combined Sen/Spec plots with harmonic mean."""
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    if subgroup_prefix:
        if subgroup_prefix == 'subgroup':
            sen_col = 'subgroup_sensitivity'
            spec_col = 'subgroup_specificity'
            title_suffix = ' (Dys)'
        elif subgroup_prefix == 'non_subgroup':
            sen_col = 'non_subgroup_sensitivity'
            spec_col = 'non_subgroup_specificity'
            title_suffix = ' (Norm)'
        else:
            sen_col = 'sensitivity'
            spec_col = 'specificity'
            title_suffix = ''
    else:
        sen_col = 'sensitivity'
        spec_col = 'specificity'
        title_suffix = ' (Overall)'
    
    if sen_col not in plot_df.columns or spec_col not in plot_df.columns:
        print(f"Warning: {sen_col} or {spec_col} not found")
        return
    
    avg_data = plot_df.groupby('Model').agg({
        sen_col: 'mean',
        spec_col: 'mean'
    }).reset_index()
    
    avg_data = avg_data.dropna(subset=[sen_col, spec_col])
    
    if avg_data.empty:
        print(f"Warning: No valid data for {sen_col} and {spec_col}")
        return
    
    avg_data['sen_spec_harmonic'] = 2 * (avg_data[sen_col] * avg_data[spec_col]) / (avg_data[sen_col] + avg_data[spec_col] + 1e-10)
    avg_data = avg_data.sort_values('sen_spec_harmonic', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(avg_data) * 0.3)))
    
    models = avg_data['Model'].values
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, avg_data[sen_col], width, 
                   label='Sensitivity', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_data[spec_col], width, 
                   label='Specificity', color='#A23B72', alpha=0.8)
    
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Sensitivity vs Specificity{title_suffix}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    ax2 = axes[1]
    colors = ['#2ECC71' if val >= 0.7 else '#F1C40F' if val >= 0.5 else '#E74C3C' 
              for val in avg_data['sen_spec_harmonic']]
    bars3 = ax2.bar(x, avg_data['sen_spec_harmonic'], color=colors, alpha=0.8)
    
    for bar, val in zip(bars3, avg_data['sen_spec_harmonic']):
        if not np.isnan(val):
            ax2.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥0.7)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (≥0.5)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Harmonic Mean (Sen/Spec)')
    ax2.set_title(f'Sen/Spec Harmonic Mean{title_suffix}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    ax3 = axes[2]
    heatmap_data = avg_data[[sen_col, spec_col]].T
    heatmap_data.columns = models
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Score'},
                linewidths=0.5, linecolor='white',
                ax=ax3, annot_kws={'fontsize': 9})
    
    row_labels = ['Sensitivity', 'Specificity']
    if subgroup_prefix == 'subgroup':
        row_labels = ['Dys-Sensitivity', 'Dys-Specificity']
    elif subgroup_prefix == 'non_subgroup':
        row_labels = ['Norm-Sensitivity', 'Norm-Specificity']
    
    ax3.set_yticklabels(row_labels, rotation=0)
    ax3.set_title(f'Sen/Spec Heatmap{title_suffix}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'sen_spec_combined{suffix}{task_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined Sen/Spec plot saved to: {output_dir / f'sen_spec_combined{suffix}{task_suffix}.png'}")
    
    avg_data.to_csv(output_dir / f'sen_spec_data{suffix}{task_suffix}.csv', index=False)
    print(f"✓ Sen/Spec data saved to: {output_dir / f'sen_spec_data{suffix}{task_suffix}.csv'}")
    
    return avg_data


def plot_subgroup_sen_spec_comprehensive(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig,
                                          task_type: str = 'classification'):
    """Create comprehensive Sen/Spec plots comparing Dys vs Norm."""
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    subgroup_sen = 'subgroup_sensitivity'
    subgroup_spec = 'subgroup_specificity'
    non_subgroup_sen = 'non_subgroup_sensitivity'
    non_subgroup_spec = 'non_subgroup_specificity'
    
    if subgroup_sen not in plot_df.columns or subgroup_spec not in plot_df.columns:
        print(f"Warning: Subgroup Sen/Spec columns not found")
        return
    
    avg_data = plot_df.groupby('Model').agg({
        subgroup_sen: 'mean',
        subgroup_spec: 'mean',
        non_subgroup_sen: 'mean',
        non_subgroup_spec: 'mean'
    }).reset_index()
    
    avg_data = avg_data.dropna(subset=[subgroup_sen, subgroup_spec, non_subgroup_sen, non_subgroup_spec])
    
    if avg_data.empty:
        print(f"Warning: No valid data for subgroup Sen/Spec after dropping NaN")
        return
    
    avg_data['dys_harmonic'] = 2 * (avg_data[subgroup_sen] * avg_data[subgroup_spec]) / (avg_data[subgroup_sen] + avg_data[subgroup_spec] + 1e-10)
    avg_data['norm_harmonic'] = 2 * (avg_data[non_subgroup_sen] * avg_data[non_subgroup_spec]) / (avg_data[non_subgroup_sen] + avg_data[non_subgroup_spec] + 1e-10)
    avg_data['harmonic_diff'] = avg_data['dys_harmonic'] - avg_data['norm_harmonic']
    avg_data = avg_data.sort_values('dys_harmonic', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = avg_data['Model'].values
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, avg_data[subgroup_sen], width, 
                   label='Dys', color='#E74C3C', alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_data[non_subgroup_sen], width, 
                   label='Norm', color='#3498DB', alpha=0.8)
    
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('Sensitivity: Dys vs Norm', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, avg_data[subgroup_spec], width, 
                   label='Dys', color='#E74C3C', alpha=0.8)
    bars4 = ax2.bar(x + width/2, avg_data[non_subgroup_spec], width, 
                   label='Norm', color='#3498DB', alpha=0.8)
    
    for bar in bars3:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    for bar in bars4:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Specificity')
    ax2.set_title('Specificity: Dys vs Norm', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    ax3 = axes[1, 0]
    bars5 = ax3.bar(x - width/2, avg_data['dys_harmonic'], width, 
                   label='Dys', color='#E74C3C', alpha=0.8)
    bars6 = ax3.bar(x + width/2, avg_data['norm_harmonic'], width, 
                   label='Norm', color='#3498DB', alpha=0.8)
    
    for bar in bars5:
        height = bar.get_height()
        if not np.isnan(height):
            ax3.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    for bar in bars6:
        height = bar.get_height()
        if not np.isnan(height):
            ax3.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Harmonic Mean (Sen/Spec)')
    ax3.set_title('Sen/Spec Harmonic Mean: Dys vs Norm', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.05)
    
    ax4 = axes[1, 1]
    
    diff_data = avg_data['harmonic_diff'].dropna()
    if diff_data.empty:
        ax4.text(0.5, 0.5, 'No valid difference data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Dys vs Norm Difference in Harmonic Mean', fontsize=13, fontweight='bold')
    else:
        colors = ['#2ECC71' if val >= 0 else '#E74C3C' for val in avg_data['harmonic_diff']]
        bars7 = ax4.bar(x, avg_data['harmonic_diff'], color=colors, alpha=0.8)
        
        for bar, val in zip(bars7, avg_data['harmonic_diff']):
            if not np.isnan(val):
                ax4.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, val),
                           xytext=(0, 3 if val >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if val >= 0 else 'top',
                           fontsize=9)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Difference (Dys - Norm)')
        ax4.set_title('Dys vs Norm Difference in Harmonic Mean', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        max_abs_diff = max(abs(avg_data['harmonic_diff'].min() or 0), abs(avg_data['harmonic_diff'].max() or 0))
        if np.isnan(max_abs_diff) or max_abs_diff == 0:
            max_abs_diff = 0.1
        ax4.set_ylim(-max_abs_diff - 0.1, max_abs_diff + 0.1)
    
    plt.tight_layout()
    task_suffix = f"_{task_type}"
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
        
        plot_data = df[df['Task'] == task_type].groupby('Model').agg({
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
        ax.set_xticklabels(plot_data['Model'], rotation=45, ha='right')
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
        
        group_cols = ['Model']
        if 'Method_Label' in task_df.columns:
            group_cols.append('Method_Label')
        elif 'Method' in task_df.columns:
            group_cols.append('Method')
        
        try:
            if len(group_cols) > 1:
                summary = task_df.groupby(group_cols).agg(agg_dict)
            else:
                summary = task_df.groupby('Model').agg(agg_dict)
        except KeyError as e:
            print(f"  Warning: Groupby failed for {task}: {e}")
            summary = task_df.groupby('Model').agg(agg_dict)
        
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
    parser.add_argument('--bootstrap-iterations', type=int, default=1000,
                        help='Number of bootstrap iterations for confidence intervals (default: 1000)')
    parser.add_argument('--no-robustness', action='store_true',
                        help='Skip robustness analysis (CI and ablation)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--subgroup', action='store_true',
                        help='Generate subgroup (Dys/Norm) analysis')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ExperimentConfig()
    config.bootstrap_iterations = args.bootstrap_iterations
    
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
    
    # ===== ROBUSTNESS ANALYSIS =====
    if not args.no_robustness:
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS ANALYSIS (Bootstrap CI & Ablation)")
        print(f"{'='*60}")
        print(f"Bootstrap iterations: {args.bootstrap_iterations}")
        
        tasks_to_process = df['Task'].unique()
        
        for task in tasks_to_process:
            print(f"\nProcessing {task} task...")
            plot_robustness_summary(df, output_dir, config, task, args.bootstrap_iterations)
    
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
                print(f"    {model}: {val:.4f}")
            
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
    print(f"  - Ablation analysis: ablation_*.csv and plots")
    print(f"  - Bootstrap iterations: {args.bootstrap_iterations}")


if __name__ == "__main__":
    main()

'''
python ~/asr_clinical/question_ensemble_fusion_aggregate_results.py --input-dir outputs-ensemble --output-dir outputs-ensemble-aggregate --subgroup --top-k 5 --bootstrap-iterations 10000 --ignore-methods mlp | tee outputs-ensemble-aggregate/log.txt
'''