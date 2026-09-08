"""
Experiment Results Aggregator - Enhanced Version
Supports BOTH classification and regression experiments with subgroup analysis (Dys/Norm).

Folder Structure:
<main_dir>/
├── classification-fusion-<model_name>/           # e.g., classification-fusion-distilroberta-base
│   ├── fusion_results/
│   │   └── leakage_safe_5fold/
│   │       ├── audio_only_aggregate_metrics.json
│   │       ├── text_only_aggregate_metrics.json
│   │       └── ...
│   ├── meta_fusion/
│   │   └── meta_fusion_metrics.json
│   └── meta_test_metrics.json
├── regression-fusion-<model_name>/               # e.g., regression-fusion-deberta-v3
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
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0-8-whitegrid')
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
    
    # Fusion methods and their JSON filenames (from leakage_safe_5fold folder)
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
    
    # Display names for fusion methods
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
    
    # Classification metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'sensitivity', 'specificity', 
        'precision', 'npv', 'f1', 'roc_auc',
        'macro_f1', 'balanced_accuracy'
    ])
    
    # Regression metrics
    regression_metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'mae', 'r2', 'spearmanr', 'pearsonr'
    ])
    
    # Subgroup metrics (for classification)
    classification_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_accuracy', 'subgroup_sensitivity', 'subgroup_specificity',
        'subgroup_precision', 'subgroup_npv', 'subgroup_f1', 'subgroup_roc_auc',
        'non_subgroup_accuracy', 'non_subgroup_sensitivity', 'non_subgroup_specificity',
        'non_subgroup_precision', 'non_subgroup_npv', 'non_subgroup_f1', 'non_subgroup_roc_auc'
    ])
    
    # Subgroup metrics (for regression)
    regression_subgroup_metrics: List[str] = field(default_factory=lambda: [
        'subgroup_rmse', 'subgroup_mae', 'subgroup_r2', 'subgroup_spearmanr', 'subgroup_pearsonr',
        'non_subgroup_rmse', 'non_subgroup_mae', 'non_subgroup_r2', 'non_subgroup_spearmanr', 'non_subgroup_pearsonr'
    ])
    
    # Metric labels
    metric_labels: Dict[str, str] = field(default_factory=lambda: {
        # Classification
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'precision': 'PPV',
        'npv': 'NPV',
        'f1': 'F1 Score',
        'roc_auc': 'AUC-ROC',
        'macro_f1': 'Macro F1',
        'balanced_accuracy': 'Balanced Accuracy',
        # Regression
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²',
        'spearmanr': "Spearman's ρ",
        'pearsonr': "Pearson's r",
        # Subgroup classification
        'subgroup_accuracy': 'Dys - Accuracy',
        'subgroup_sensitivity': 'Dys - Sensitivity',
        'subgroup_specificity': 'Dys - Specificity',
        'subgroup_precision': 'Dys - PPV',
        'subgroup_npv': 'Dys - NPV',
        'subgroup_f1': 'Dys - F1',
        'subgroup_roc_auc': 'Dys - AUC-ROC',
        'non_subgroup_accuracy': 'Norm - Accuracy',
        'non_subgroup_sensitivity': 'Norm - Sensitivity',
        'non_subgroup_specificity': 'Norm - Specificity',
        'non_subgroup_precision': 'Norm - PPV',
        'non_subgroup_npv': 'Norm - NPV',
        'non_subgroup_f1': 'Norm - F1',
        'non_subgroup_roc_auc': 'Norm - AUC-ROC',
        # Subgroup regression
        'subgroup_rmse': 'Dys - RMSE',
        'subgroup_mae': 'Dys - MAE',
        'subgroup_r2': 'Dys - R²',
        'subgroup_spearmanr': 'Dys - Spearman\'s ρ',
        'subgroup_pearsonr': 'Dys - Pearson\'s r',
        'non_subgroup_rmse': 'Norm - RMSE',
        'non_subgroup_mae': 'Norm - MAE',
        'non_subgroup_r2': 'Norm - R²',
        'non_subgroup_spearmanr': 'Norm - Spearman\'s ρ',
        'non_subgroup_pearsonr': 'Norm - Pearson\'s r'
    })
    
    # Ranking metrics
    ranking_metric_classification: str = 'roc_auc'
    ranking_metric_regression: str = 'r2'


# =======================================================================
#  FOLDER PARSING FUNCTIONS
# =======================================================================

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
    """
    Parse folder name for classification-fusion-* or regression-fusion-* pattern.
    
    Args:
        folder_name: Name of the folder
    
    Returns:
        Dict with task, model, and other metadata
    """
    result = {
        'task': 'unknown',
        'model': folder_name,
        'full_name': folder_name,
        'size': 'Unknown',
        'family': 'Other'
    }
    
    # Detect task
    if folder_name.startswith('classification'):
        result['task'] = 'classification'
        prefix = 'classification-fusion-'
        if folder_name.startswith(prefix):
            model_part = folder_name[len(prefix):]
        elif folder_name.startswith('classification-'):
            model_part = folder_name[len('classification-'):]
        else:
            model_part = folder_name
    elif folder_name.startswith('regression'):
        result['task'] = 'regression'
        prefix = 'regression-fusion-'
        if folder_name.startswith(prefix):
            model_part = folder_name[len(prefix):]
        elif folder_name.startswith('regression-'):
            model_part = folder_name[len('regression-'):]
        else:
            model_part = folder_name
    else:
        model_part = folder_name
    
    result['model'] = model_part
    
    # Detect model size and family
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
    """
    Extract metrics from result dictionary.
    Handles both overall metrics and subgroup (Dys/Norm) metrics.
    """
    extracted = {}
    
    if not isinstance(result, dict):
        return extracted
    
    # Extract overall metrics
    if 'all' in result and isinstance(result['all'], dict):
        # Extract from 'all' key
        for key, value in result['all'].items():
            if isinstance(value, (int, float)):
                extracted[key] = value
    else:
        # Direct extraction
        for key, value in result.items():
            if isinstance(value, (int, float)):
                # Skip internal/verbose fields
                if key not in ['threshold', 'threshold_used', 'k_neighbors', 'avg_best_k']:
                    extracted[key] = value
    
    # Extract subgroup metrics (Dys)
    if 'subgroup' in result and isinstance(result['subgroup'], dict):
        for key, value in result['subgroup'].items():
            if isinstance(value, (int, float)):
                extracted[f'subgroup_{key}'] = value
    
    # Extract non-subgroup metrics (Norm)
    if 'non_subgroup' in result and isinstance(result['non_subgroup'], dict):
        for key, value in result['non_subgroup'].items():
            if isinstance(value, (int, float)):
                extracted[f'non_subgroup_{key}'] = value
    
    # Also check for nested structure in 'all' with subgroup info
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
    """
    Discover experiments by scanning folders matching classification-fusion-* or regression-fusion-* pattern.
    
    Args:
        base_dir: Base directory containing experiment folders
        task_type: 'classification', 'regression', or 'all'
        config: ExperimentConfig instance
    
    Returns:
        Dictionary of discovered experiments
    """
    if config is None:
        config = ExperimentConfig()
    
    experiments = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n{'='*60}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*60}")
    print(f"Task type filter: {task_type}")
    
    # Determine patterns based on task type
    if task_type == 'classification':
        patterns = ['classification-fusion-*']
    elif task_type == 'regression':
        patterns = ['regression-fusion-*']
    else:
        patterns = ['classification-fusion-*', 'regression-fusion-*']
    
    all_folders = []
    for pattern in patterns:
        all_folders.extend(list(base_dir.glob(pattern)))
    
    # Also check for folders without 'fusion' in name
    if task_type in ['classification', 'all']:
        all_folders.extend([f for f in base_dir.glob("classification-*") 
                           if 'fusion' not in f.name and f not in all_folders])
    if task_type in ['regression', 'all']:
        all_folders.extend([f for f in base_dir.glob("regression-*") 
                           if 'fusion' not in f.name and f not in all_folders])
    
    all_folders = list(set(all_folders))
    
    print(f"\nFound {len(all_folders)} potential experiment folders")
    
    # Process each folder
    for folder_path in all_folders:
        folder_name = folder_path.name
        
        # Parse folder name
        parsed = parse_folder_name(folder_name)
        model_name = parsed['model']
        task = parsed['task']
        size = parsed['size']
        family = parsed['family']
        
        # Skip if task doesn't match filter
        if task_type != 'all' and task != task_type:
            continue
        
        # Skip if model name is empty or generic
        if model_name in ['', 'fusion', 'classification', 'regression']:
            continue
        
        # Create a composite key
        model_key = f"{task}_{model_name}"
        
        print(f"\nProcessing: {folder_name}")
        print(f"  Task: {task}")
        print(f"  Model: {model_name}")
        print(f"  Size: {size}, Family: {family}")
        
        # Check fusion_results subdirectory
        fusion_dir = folder_path / 'fusion_results' / 'leakage_safe_5fold'
        meta_fusion_dir = folder_path / 'meta_fusion'
        main_metrics = folder_path / 'meta_test_metrics.json'
        cv_metrics = folder_path / 'cv_aggregate_metrics.json'
        
        # Look for fusion method metrics
        found_fusion = False
        for method_key, filename in config.fusion_methods.items():
            metrics_file = fusion_dir / filename
            
            # Also try alternative filenames
            if not metrics_file.exists():
                # Try without _aggregate_metrics
                alt_name = filename.replace('_aggregate_metrics.json', '_metrics.json')
                alt_file = fusion_dir / alt_name
                if alt_file.exists():
                    metrics_file = alt_file
                else:
                    # Try without any suffix
                    base_name = method_key.replace('_', '')
                    for alt in [f"{method_key}.json", f"{method_key}_metrics.json"]:
                        alt_file = fusion_dir / alt
                        if alt_file.exists():
                            metrics_file = alt_file
                            break
            
            if metrics_file.exists():
                experiments[model_key][model_name][method_key] = {
                    'task': task,
                    'model': model_name,
                    'size': size,
                    'family': family,
                    'metrics_file': metrics_file,
                    'dir': folder_path,
                    'fusion_dir': fusion_dir,
                    'source': 'fusion_results'
                }
                found_fusion = True
                print(f"    ✓ Found: {method_key}")
        
        # Check for meta_fusion results
        meta_fusion_file = meta_fusion_dir / 'meta_fusion_metrics.json'
        if meta_fusion_file.exists():
            experiments[model_key][model_name]['meta_fusion'] = {
                'task': task,
                'model': model_name,
                'size': size,
                'family': family,
                'metrics_file': meta_fusion_file,
                'dir': folder_path,
                'fusion_dir': meta_fusion_dir,
                'source': 'meta_fusion'
            }
            print(f"    ✓ Found meta_fusion results")
        
        # Look for main results
        if main_metrics.exists():
            experiments[model_key][model_name]['main'] = {
                'task': task,
                'model': model_name,
                'size': size,
                'family': family,
                'metrics_file': main_metrics,
                'dir': folder_path,
                'fusion_dir': None,
                'source': 'main'
            }
            print(f"    ✓ Found main results")
        
        # Look for CV results
        if cv_metrics.exists():
            experiments[model_key][model_name]['cv'] = {
                'task': task,
                'model': model_name,
                'size': size,
                'family': family,
                'metrics_file': cv_metrics,
                'dir': folder_path,
                'fusion_dir': None,
                'source': 'cv'
            }
            print(f"    ✓ Found CV results")
        
        if not found_fusion and not main_metrics.exists():
            print(f"    ⚠ No metrics found")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    
    total_methods = 0
    classification_models = 0
    regression_models = 0
    
    for key, model_data in experiments.items():
        task, model = key.split('_', 1)
        if task == 'classification':
            classification_models += 1
        else:
            regression_models += 1
        
        print(f"\nTask: {task}, Model: {model}")
        method_count = 0
        for method_key in model_data.keys():
            if method_key not in ['main', 'cv', 'meta_fusion']:
                method_count += 1
        print(f"  {method_count} fusion methods")
        if 'meta_fusion' in model_data:
            print(f"  meta_fusion results")
        if 'main' in model_data:
            print(f"  main results")
        if 'cv' in model_data:
            print(f"  CV results")
        total_methods += method_count
    
    print(f"\nTotal: {len(experiments)} models ({classification_models} classification, {regression_models} regression)")
    print(f"Total fusion method results: {total_methods}")
    
    return dict(experiments)


# =======================================================================
#  DATA AGGREGATION FUNCTIONS
# =======================================================================

def aggregate_experiment_results(experiments: Dict, config: ExperimentConfig) -> pd.DataFrame:
    """Aggregate all experiment results into a single DataFrame."""
    rows = []
    
    for key, model_data in experiments.items():
        # Parse key to get task and model
        parts = key.split('_', 1)
        task = parts[0] if len(parts) > 1 else 'classification'
        model_name = parts[1] if len(parts) > 1 else key
        
        for method_key, method_data in model_data.items():
            # Skip main and cv results for fusion method aggregation
            if method_key in ['main', 'cv', 'meta_fusion']:
                continue
            
            metrics = load_metrics_file(method_data.get('metrics_file'))
            if metrics is None:
                continue
            
            extracted = extract_metrics_from_result(metrics, task)
            
            method_display = config.method_display_names.get(method_key, method_key)
            size = method_data.get('size', 'Unknown')
            family = method_data.get('family', 'Other')
            
            row = {
                'Task': task,
                'Model': model_name,
                'Size': size,
                'Family': family,
                'Method': method_key,
                'Method_Label': method_display,
                'Source': method_data.get('source', 'fusion_results')
            }
            
            # Add metrics based on task
            if task == 'classification':
                all_metrics = config.classification_metrics + config.classification_subgroup_metrics
            else:
                all_metrics = config.regression_metrics + config.regression_subgroup_metrics
            
            for metric in all_metrics:
                # Try to get from extracted metrics
                row[metric] = extracted.get(metric, None)
                
                # If not found, try direct from metrics
                if row[metric] is None and metric in metrics:
                    row[metric] = metrics.get(metric, None)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert metrics to numeric
    all_metrics = (config.classification_metrics + config.classification_subgroup_metrics + 
                   config.regression_metrics + config.regression_subgroup_metrics)
    for metric in all_metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


def aggregate_main_results(experiments: Dict, config: ExperimentConfig) -> pd.DataFrame:
    """Aggregate main results (meta_test_metrics.json) for each run."""
    rows = []
    
    for key, model_data in experiments.items():
        parts = key.split('_', 1)
        task = parts[0] if len(parts) > 1 else 'classification'
        model_name = parts[1] if len(parts) > 1 else key
        
        for method_key, method_data in model_data.items():
            if method_key not in ['main', 'cv', 'meta_fusion']:
                continue
            
            metrics = load_metrics_file(method_data.get('metrics_file'))
            if metrics is None:
                continue
            
            extracted = extract_metrics_from_result(metrics, task)
            
            size = method_data.get('size', 'Unknown')
            family = method_data.get('family', 'Other')
            source = method_data.get('source', 'main')
            
            row = {
                'Task': task,
                'Model': model_name,
                'Size': size,
                'Family': family,
                'Source': source,
                'Source_Label': source.upper(),
                'Best_K': extracted.get('avg_best_k', extracted.get('best_k', None)),
                'Selected_Questions': str(extracted.get('selected_questions', []))
            }
            
            # Add metrics based on task
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


# =======================================================================
#  VISUALIZATION FUNCTIONS
# =======================================================================

def get_task_metrics(df: pd.DataFrame, config: ExperimentConfig, task: str) -> List[str]:
    """Get appropriate metrics for a task."""
    if task == 'classification':
        return config.classification_metrics
    else:
        return config.regression_metrics


def get_subgroup_metrics(df: pd.DataFrame, config: ExperimentConfig, task: str) -> List[str]:
    """Get appropriate subgroup metrics for a task."""
    if task == 'classification':
        return config.classification_subgroup_metrics
    else:
        return config.regression_subgroup_metrics


def plot_method_comparison(df: pd.DataFrame, metric: str, output_dir: Path, 
                           config: ExperimentConfig, title: str = None, 
                           subgroup_prefix: str = None, task_type: str = 'classification'):
    """Create grouped bar chart comparing methods across models."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for {metric}")
        return
    
    # Filter data for specific task
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    # Group by Model and Method
    pivot = plot_df.pivot_table(
        index='Model',
        columns='Method_Label',
        values=metric,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"Warning: No data for method_comparison_{metric}")
        return
    
    # Determine sorting direction
    lower_is_better = metric in ['rmse', 'mae']
    
    # Sort models by best method
    if lower_is_better:
        best_vals = pivot.min(axis=1)
    else:
        best_vals = pivot.max(axis=1)
    pivot = pivot.loc[best_vals.sort_values(ascending=lower_is_better).index]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot.index) * 0.4)))
    
    # Create grouped bar chart
    pivot.plot(kind='barh', ax=ax, width=0.8, colormap='viridis')
    
    metric_label = config.metric_labels.get(metric, metric.upper())
    title_text = title or f'{metric_label} by Model and Fusion Method ({task_type.title()})'
    if subgroup_prefix:
        group_name = 'Dys' if 'subgroup' in subgroup_prefix else 'Norm'
        title_text = f'{group_name} - {metric_label} by Model and Fusion Method ({task_type.title()})'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Model')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    suffix = f"_{subgroup_prefix}" if subgroup_prefix else ""
    task_suffix = f"_{task_type}"
    plt.savefig(output_dir / f'method_comparison_{metric}{suffix}{task_suffix}.png', dpi=300)
    plt.close()
    print(f"✓ Method comparison saved to: {output_dir / f'method_comparison_{metric}{suffix}{task_suffix}.png'}")


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig,
                 subgroup_prefix: str = None, task_type: str = 'classification'):
    """Create heatmap comparing methods across models."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for heatmap_{metric}")
        return
    
    # Filter data for specific task
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    # Create pivot table
    pivot = plot_df.pivot_table(
        index='Method_Label',
        columns='Model',
        values=metric,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"Warning: Empty pivot for heatmap_{metric}")
        return
    
    # Drop columns with all NaN
    pivot = pivot.dropna(axis=1, how='all')
    
    if pivot.empty:
        print(f"Warning: No data after dropping NaN for heatmap_{metric}")
        return
    
    # Determine colormap
    lower_is_better = metric in ['rmse', 'mae']
    if lower_is_better:
        cmap = 'RdYlGn_r'  # Lower is better
    else:
        cmap = 'RdYlGn_r'  # Higher is better
    
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
    plt.savefig(output_dir / f'heatmap_{metric}{suffix}{task_suffix}.png', dpi=300)
    plt.close()
    print(f"✓ Heatmap saved to: {output_dir / f'heatmap_{metric}{suffix}{task_suffix}.png'}")


def plot_subgroup_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig, 
                             task_type: str = 'classification'):
    """Create plots comparing Dys vs Norm performance for a specific task."""
    # Filter data for specific task
    plot_df = df[df['Task'] == task_type].copy()
    if plot_df.empty:
        print(f"Warning: No data for task {task_type}")
        return
    
    # Determine metrics to compare based on task
    if task_type == 'classification':
        metrics_to_compare = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']
        prefix = ''
    else:
        metrics_to_compare = ['rmse', 'mae', 'r2', 'spearmanr', 'pearsonr']
        prefix = ''
    
    # Create grouped bar chart for subgroup comparison
    for metric in metrics_to_compare:
        subgroup_metric = f'subgroup_{metric}'
        non_subgroup_metric = f'non_subgroup_{metric}'
        
        if subgroup_metric not in plot_df.columns or non_subgroup_metric not in plot_df.columns:
            continue
        
        # Prepare data
        plot_data = plot_df[['Model', 'Method_Label', subgroup_metric, non_subgroup_metric]].dropna()
        if plot_data.empty:
            continue
        
        # Melt for plotting
        plot_data_melted = plot_data.melt(
            id_vars=['Model', 'Method_Label'],
            value_vars=[subgroup_metric, non_subgroup_metric],
            var_name='Group',
            value_name='Score'
        )
        plot_data_melted['Group'] = plot_data_melted['Group'].map({
            subgroup_metric: 'Dys (Subgroup)',
            non_subgroup_metric: 'Norm (Non-Subgroup)'
        })
        
        # Calculate averages
        avg_data = plot_data_melted.groupby(['Model', 'Group'])['Score'].mean().reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        pivot_avg = avg_data.pivot(index='Model', columns='Group', values='Score')
        
        # Sort by Dys performance
        if metric in ['rmse', 'mae']:
            pivot_avg = pivot_avg.sort_values('Dys (Subgroup)', ascending=True)
        else:
            pivot_avg = pivot_avg.sort_values('Dys (Subgroup)', ascending=False)
        
        colors = ['#E74C3C', '#3498DB']  # Red for Dys, Blue for Norm
        pivot_avg.plot(kind='bar', ax=ax, width=0.8, color=colors)
        
        metric_label = config.metric_labels.get(metric, metric.upper())
        ax.set_title(f'{metric_label}: Dys vs Norm Comparison Across Models ({task_type.title()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_label)
        ax.legend(title='Group')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        task_suffix = f"_{task_type}"
        plt.savefig(output_dir / f'subgroup_comparison_{metric}{task_suffix}.png', dpi=300)
        plt.close()
        print(f"✓ Subgroup comparison saved to: {output_dir / f'subgroup_comparison_{metric}{task_suffix}.png'}")


def create_summary_table(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Create summary table with mean and std for each model-method."""
    # Determine metrics based on task
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
        
        # Group and aggregate
        agg_dict = {}
        for metric in all_metrics:
            agg_dict[metric] = ['mean', 'std', 'count']
        
        summary = task_df.groupby(['Model', 'Method_Label']).agg(agg_dict)
        summary = summary.round(4)
        
        # Save as CSV
        summary.to_csv(output_dir / f'summary_table_{task}.csv')
        print(f"✓ Summary table saved to: {output_dir / f'summary_table_{task}.csv'}")
        
        # Create flattened version
        flat_summary = summary.copy()
        flat_summary.columns = [f'{col[0]}_{col[1]}' for col in flat_summary.columns]
        flat_summary.reset_index().to_csv(output_dir / f'summary_table_flat_{task}.csv', index=False)
    
    return


# =======================================================================
#  MAIN FUNCTION
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flexible Experiment Results Aggregator - Supports Classification & Regression'
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
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to include in figures')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--subgroup', action='store_true',
                        help='Generate subgroup (Dys/Norm) analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Discover experiments
    base_dir = Path(args.input_dir)
    experiments = discover_experiments(base_dir, args.task, config)
    
    if not experiments:
        print("\n" + "="*60)
        print("ERROR: No experiments found!")
        print("="*60)
        print(f"Base directory: {base_dir}")
        print("\nExpected folder structure:")
        print("  classification-fusion-<model_name>/")
        print("  regression-fusion-<model_name>/")
        print("    └── fusion_results/leakage_safe_5fold/")
        return
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    df = aggregate_experiment_results(experiments, config)
    main_df = aggregate_main_results(experiments, config)
    
    # Apply filters
    if args.task != 'all':
        df = df[df['Task'] == args.task]
        main_df = main_df[main_df['Task'] == args.task]
    
    if args.models:
        df = df[df['Model'].isin(args.models)]
        main_df = main_df[main_df['Model'].isin(args.models)]
    
    if args.methods:
        df = df[df['Method'].isin(args.methods)]
    
    print(f"\nAggregated {len(df)} fusion method results")
    print(f"Aggregated {len(main_df)} main results")
    
    if df.empty:
        print("\n" + "="*60)
        print("ERROR: No data found after filtering!")
        print("="*60)
        return
    
    # Print summary
    print("\nTasks found:", df['Task'].unique().tolist())
    print("Models found:", df['Model'].unique().tolist())
    print("Model sizes:", df['Size'].unique().tolist())
    print("Model families:", df['Family'].unique().tolist())
    print("Methods found:", df['Method_Label'].unique().tolist())
    
    # Save results
    main_df.to_csv(output_dir / 'main_results.csv', index=False)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate summary tables
    print(f"\n{'='*60}")
    print(f"GENERATING SUMMARY TABLES")
    print(f"{'='*60}")
    
    create_summary_table(df, output_dir, config)
    
    # Generate figures
    if not args.no_plots:
        print(f"\n{'='*60}")
        print(f"GENERATING FIGURES")
        print(f"{'='*60}")
        
        # Determine tasks to process
        tasks_to_process = df['Task'].unique()
        
        for task in tasks_to_process:
            print(f"\nProcessing {task} task...")
            
            # Determine metrics for this task
            if task == 'classification':
                default_metrics = ['accuracy', 'sensitivity', 'specificity', 'roc_auc', 'f1']
                all_metrics = config.classification_metrics + config.classification_subgroup_metrics
            else:
                default_metrics = ['rmse', 'mae', 'r2']
                all_metrics = config.regression_metrics + config.regression_subgroup_metrics
            
            if args.metrics:
                metrics_to_plot = [m for m in args.metrics if m in all_metrics]
            else:
                metrics_to_plot = [m for m in default_metrics if m in df.columns and df[m].notna().any()]
            
            # Plot main metrics
            for metric in metrics_to_plot:
                if metric in df.columns and df[df['Task'] == task][metric].notna().any():
                    plot_heatmap(df, metric, output_dir, config, task_type=task)
                    plot_method_comparison(df, metric, output_dir, config, task_type=task)
            
            # Plot subgroup metrics
            if args.subgroup:
                if task == 'classification':
                    subgroup_metrics = ['subgroup_roc_auc', 'subgroup_accuracy', 'subgroup_f1']
                    if args.metrics:
                        subgroup_metrics = [m for m in args.metrics if m.startswith(('subgroup_', 'non_subgroup_'))]
                else:
                    subgroup_metrics = ['subgroup_r2', 'subgroup_rmse', 'subgroup_mae']
                    if args.metrics:
                        subgroup_metrics = [m for m in args.metrics if m.startswith(('subgroup_', 'non_subgroup_'))]
                
                # Plot subgroup-specific metrics
                for metric in subgroup_metrics:
                    if metric in df.columns and df[df['Task'] == task][metric].notna().any():
                        # Determine if it's subgroup or non-subgroup
                        prefix = 'subgroup' if 'subgroup' in metric else 'non_subgroup'
                        plot_heatmap(df, metric, output_dir, config, subgroup_prefix=prefix, task_type=task)
                        plot_method_comparison(df, metric, output_dir, config, subgroup_prefix=prefix, task_type=task)
                
                # Generate Dys vs Norm comparison plots
                plot_subgroup_comparison(df, output_dir, config, task_type=task)
    
    # Print summary statistics
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
            
            if ranking_metric in ['rmse', 'mae']:
                best_by_method = best_by_method.sort_values(ascending=True)
            else:
                best_by_method = best_by_method.sort_values(ascending=False)
            
            for method, val in best_by_method.head(5).items():
                print(f"    {method}: {val:.4f}")
            
            print(f"\n  Best by Model:")
            best_by_model = task_df.groupby('Model')[ranking_metric].mean()
            if ranking_metric in ['rmse', 'mae']:
                best_by_model = best_by_model.sort_values(ascending=True)
            else:
                best_by_model = best_by_model.sort_values(ascending=False)
            for model, val in best_by_model.head(5).items():
                print(f"    {model}: {val:.4f}")
    
    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()