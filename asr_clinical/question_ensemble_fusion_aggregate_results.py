"""
Experiment Results Aggregator - Flexible Version
Supports BOTH classification and regression experiments with customizable folder patterns.

Folder Structure (flexible):
<main_dir>/
├── <pattern_1>_<model_name>/           # e.g., classification-bal-fusion-distilroberta-base
│   ├── fusion_results/
│   │   └── leakage_safe_5fold/
│   │       ├── audio_only_aggregate_metrics.json
│   │       ├── text_only_aggregate_metrics.json
│   │       ├── early_aggregate_metrics.json
│   │       └── ...
│   ├── cv_aggregate_metrics.json
│   └── meta_test_metrics.json
├── <pattern_2>_<model_name>/           # e.g., regression-fusion-deberta-v3
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
    
    # Alternative method names for matching
    method_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        'audio_only': ['audio', 'audio_only'],
        'text_only': ['text', 'text_only'],
        'early': ['early', 'early_fusion'],
        'late': ['late', 'late_fusion'],
        'confidence': ['confidence', 'confidence_weighted'],
        'interaction': ['interaction', 'interaction_stacking'],
        'moe': ['moe', 'mixture_of_experts'],
        'mlp': ['mlp', 'mlp_early_fusion'],
        'stacking': ['stacking', 'model_based_fusion', 'model_based_stacking'],
        'cca': ['cca'],
        'dynamic': ['dynamic']
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
    
    # Strategy display names
    strategy_display_names: Dict[str, str] = field(default_factory=lambda: {
        'balanced': 'Balanced',
        'unbalanced': 'Unbalanced',
        'focal': 'Focal Loss',
        'weighted': 'Weighted'
    })
    
    # Classification metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'sensitivity', 'specificity', 
        'precision', 'npv', 'f1', 'roc_auc',
        'macro_f1', 'balanced_accuracy'
    ])
    
    # Regression metrics
    regression_metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'mae', 'r2'
    ])
    
    # Metric labels
    metric_labels: Dict[str, str] = field(default_factory=lambda: {
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'precision': 'PPV',
        'npv': 'NPV',
        'f1': 'F1 Score',
        'roc_auc': 'AUC-ROC',
        'macro_f1': 'Macro F1',
        'balanced_accuracy': 'Balanced Accuracy',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²'
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
        return 'Clinical'
    elif 'biomed' in model_lower:
        return 'BioMed'
    elif 'scibert' in model_lower:
        return 'SciBERT'
    elif 'legal' in model_lower:
        return 'Legal'
    else:
        return 'Other'


def parse_folder_name(folder_name: str, patterns: List[str]) -> Dict[str, str]:
    """
    Parse folder name using provided patterns.
    
    Args:
        folder_name: Name of the folder
        patterns: List of patterns to match (e.g., ['classification-bal-fusion', 'regression-fusion'])
    
    Returns:
        Dict with task, strategy, model, and other metadata
    """
    result = {
        'task': 'unknown',
        'strategy': 'unknown',
        'model': folder_name,
        'full_name': folder_name,
        'size': 'Unknown',
        'family': 'Other'
    }
    
    # Try to detect task from folder name
    if 'classification' in folder_name or 'class' in folder_name:
        result['task'] = 'classification'
    elif 'regression' in folder_name or 'regress' in folder_name:
        result['task'] = 'regression'
    
    # Try to detect strategy
    if 'bal' in folder_name or 'balanced' in folder_name:
        result['strategy'] = 'balanced'
    elif 'focal' in folder_name:
        result['strategy'] = 'focal'
    elif 'weighted' in folder_name:
        result['strategy'] = 'weighted'
    elif 'unbal' in folder_name or 'unbalanced' in folder_name:
        result['strategy'] = 'unbalanced'
    
    # Extract model name using patterns
    for pattern in patterns:
        # Remove pattern prefix from folder name
        if folder_name.startswith(pattern):
            model_part = folder_name[len(pattern):]
            # Remove any trailing separators
            model_part = model_part.lstrip('-_')
            
            # Try to extract the actual model name
            # Common patterns: model_name, model_name-suffix, model_name_strategy
            # Remove common suffixes
            model_part = re.sub(r'[-_](balanced|unbalanced|focal|weighted|no_balance)$', '', model_part)
            model_part = re.sub(r'[-_](base|large|small|medium)$', '', model_part)
            
            result['model'] = model_part
            break
    
    # If model not extracted, try to guess
    if result['model'] == folder_name:
        # Try to remove common prefixes
        cleaned = folder_name
        for prefix in ['classification-', 'regression-', 'bal-', 'fusion-', 'focal-']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        # Remove strategy suffixes
        cleaned = re.sub(r'[-_](balanced|unbalanced|focal|weighted)$', '', cleaned)
        result['model'] = cleaned
    
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
    """Extract relevant metrics from result dictionary based on task."""
    extracted = {}
    
    if not isinstance(result, dict):
        return extracted
    
    # Try to find metrics in nested structure
    def extract_from_dict(d: Dict, prefix: str = ''):
        for key, value in d.items():
            if isinstance(value, dict):
                extract_from_dict(value, f"{prefix}{key}.")
            elif isinstance(value, (int, float)):
                # Skip internal/verbose fields
                if key not in ['threshold', 'threshold_used', 'k_neighbors']:
                    extracted[f"{prefix}{key}"] = value
    
    # If it's a direct metrics dict
    if 'all' in result and isinstance(result['all'], dict):
        # This is a subgroup result
        extract_from_dict(result['all'])
        # Also extract subgroup metrics if available
        for subgroup in ['subgroup', 'non_subgroup']:
            if subgroup in result and isinstance(result[subgroup], dict):
                for key, value in result[subgroup].items():
                    if isinstance(value, (int, float)):
                        extracted[f"{subgroup}_{key}"] = value
    else:
        extract_from_dict(result)
    
    # Ensure standard metrics are present
    if task == 'classification':
        standard_metrics = [
            'accuracy', 'sensitivity', 'specificity', 
            'precision', 'npv', 'f1', 'roc_auc',
            'macro_f1', 'balanced_accuracy'
        ]
    else:
        standard_metrics = ['rmse', 'mae', 'r2']
    
    for metric in standard_metrics:
        # Try different possible keys
        found = False
        for key in [metric, f"all_{metric}", f"test_{metric}", f"cv_{metric}"]:
            if key in extracted:
                extracted[metric] = extracted[key]
                found = True
                break
        if not found:
            extracted[metric] = None
    
    return extracted


def get_ranking_metric(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Get the appropriate ranking metric based on task."""
    if df.empty:
        return config.ranking_metric_classification
    
    tasks = df['Task'].unique()
    if len(tasks) == 0:
        return config.ranking_metric_classification
    
    # Check if we have regression tasks
    has_regression = 'regression' in tasks
    has_classification = 'classification' in tasks
    
    if has_regression and not has_classification:
        return config.ranking_metric_regression
    elif has_classification and not has_regression:
        return config.ranking_metric_classification
    else:
        # Mixed tasks - use available metrics
        available = []
        for metric in ['roc_auc', 'r2', 'accuracy', 'rmse']:
            if metric in df.columns and df[metric].notna().any():
                available.append(metric)
        if 'roc_auc' in available:
            return 'roc_auc'
        elif 'r2' in available:
            return 'r2'
        elif 'accuracy' in available:
            return 'accuracy'
        else:
            return 'rmse'


# =======================================================================
#  EXPERIMENT DISCOVERY
# =======================================================================

def discover_experiments(base_dir: Path, patterns: List[str], 
                         fusion_subdir: str = 'leakage_safe_5fold',
                         config: ExperimentConfig = None) -> Dict:
    """
    Discover experiments by scanning folders matching provided patterns.
    
    Args:
        base_dir: Base directory containing experiment folders
        patterns: List of patterns to match folder names
        fusion_subdir: Subdirectory under fusion_results to look for metrics
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
    print(f"\nUsing patterns: {patterns}")
    print(f"Fusion subdirectory: {fusion_subdir}")
    
    # Find all folders matching patterns
    all_folders = []
    for pattern in patterns:
        # Pattern can include wildcards
        if '*' in pattern:
            all_folders.extend(list(base_dir.glob(pattern)))
        else:
            # Exact pattern with wildcard at end
            all_folders.extend(list(base_dir.glob(f"{pattern}*")))
    
    all_folders = list(set(all_folders))
    
    print(f"\nFound {len(all_folders)} potential experiment folders")
    
    # Process each folder
    for folder_path in all_folders:
        folder_name = folder_path.name
        
        # Parse folder name
        parsed = parse_folder_name(folder_name, patterns)
        model_name = parsed['model']
        strategy_key = parsed['strategy']
        task = parsed['task']
        size = parsed['size']
        family = parsed['family']
        
        # Skip if model name is empty or generic
        if model_name in ['', 'fusion', 'bal', 'focal', 'classification', 'regression']:
            continue
        
        # Create a composite key
        model_strategy_key = f"{task}_{model_name}"
        strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
        
        print(f"\nProcessing: {folder_name}")
        print(f"  Task: {task}")
        print(f"  Model: {model_name}")
        print(f"  Strategy: {strategy_key} ({strategy_display})")
        print(f"  Size: {size}, Family: {family}")
        
        # Check fusion_results subdirectory
        fusion_dir = folder_path / 'fusion_results' / fusion_subdir
        main_metrics = folder_path / 'meta_test_metrics.json'
        
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
                experiments[model_strategy_key][strategy_key][method_key] = {
                    'task': task,
                    'model': model_name,
                    'size': size,
                    'family': family,
                    'strategy_display': strategy_display,
                    'metrics_file': metrics_file,
                    'dir': folder_path,
                    'fusion_dir': fusion_dir
                }
                found_fusion = True
                print(f"    ✓ Found: {method_key}")
        
        # Look for main results
        if main_metrics.exists():
            experiments[model_strategy_key][strategy_key]['main'] = {
                'task': task,
                'model': model_name,
                'size': size,
                'family': family,
                'strategy_display': strategy_display,
                'metrics_file': main_metrics,
                'dir': folder_path,
                'fusion_dir': None
            }
            print(f"    ✓ Found main results")
        
        if not found_fusion and not main_metrics.exists():
            print(f"    ⚠ No metrics found in {fusion_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    
    total_methods = 0
    for key, strategies in experiments.items():
        task, model = key.split('_', 1)
        print(f"\nTask: {task}, Model: {model}")
        for strategy, methods in strategies.items():
            strat_display = config.strategy_display_names.get(strategy, strategy)
            method_count = len([m for m in methods if m != 'main'])
            print(f"  {strat_display}: {method_count} methods")
            total_methods += method_count
    
    print(f"\nTotal: {len(experiments)} model-strategy combinations, {total_methods} fusion method results")
    
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
        
        for strategy_key, strategy_data in model_data.items():
            for method_key, method_data in strategy_data.items():
                if method_key == 'main':
                    continue
                
                metrics = load_metrics_file(method_data.get('metrics_file'))
                if metrics is None:
                    continue
                
                extracted = extract_metrics_from_result(metrics, task)
                
                strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
                method_display = config.method_display_names.get(method_key, method_key)
                size = method_data.get('size', 'Unknown')
                family = method_data.get('family', 'Other')
                
                row = {
                    'Task': task,
                    'Model': model_name,
                    'Size': size,
                    'Family': family,
                    'Strategy': strategy_key,
                    'Strategy_Label': strategy_display,
                    'Method': method_key,
                    'Method_Label': method_display,
                }
                
                # Add metrics based on task
                if task == 'classification':
                    metrics_to_add = config.classification_metrics
                else:
                    metrics_to_add = config.regression_metrics
                
                for metric in metrics_to_add:
                    # Try to get from extracted metrics
                    row[metric] = extracted.get(metric, None)
                    
                    # If not found, try direct from metrics
                    if row[metric] is None and metric in metrics:
                        row[metric] = metrics.get(metric, None)
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert metrics to numeric
    all_metrics = config.classification_metrics + config.regression_metrics
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
        
        for strategy_key, strategy_data in model_data.items():
            if 'main' not in strategy_data:
                continue
            
            main_data = strategy_data['main']
            metrics = load_metrics_file(main_data.get('metrics_file'))
            if metrics is None:
                continue
            
            extracted = extract_metrics_from_result(metrics, task)
            
            strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
            size = main_data.get('size', 'Unknown')
            family = main_data.get('family', 'Other')
            
            row = {
                'Task': task,
                'Model': model_name,
                'Size': size,
                'Family': family,
                'Strategy': strategy_key,
                'Strategy_Label': strategy_display,
                'Best_K': extracted.get('avg_best_k', extracted.get('best_k', None)),
                'Selected_Questions': str(extracted.get('selected_questions', []))
            }
            
            if task == 'classification':
                for metric in config.classification_metrics:
                    row[metric] = extracted.get(metric, None)
                    if row[metric] is None and metric in metrics:
                        row[metric] = metrics.get(metric, None)
            else:
                for metric in config.regression_metrics:
                    row[metric] = extracted.get(metric, None)
                    if row[metric] is None and metric in metrics:
                        row[metric] = metrics.get(metric, None)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    all_metrics = config.classification_metrics + config.regression_metrics
    for metric in all_metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


# =======================================================================
#  VISUALIZATION FUNCTIONS
# =======================================================================

def plot_method_comparison(df: pd.DataFrame, metric: str, output_dir: Path, 
                           config: ExperimentConfig, title: str = None):
    """Create grouped bar chart comparing methods across models."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for {metric}")
        return
    
    lower_is_better = metric in ['rmse', 'mae']
    
    # Group by Model and Method
    pivot = df.pivot_table(
        index='Model',
        columns='Method_Label',
        values=metric,
        aggfunc='mean'
    )
    
    if pivot.empty:
        print(f"Warning: No data for method_comparison_{metric}")
        return
    
    # Sort models by best method
    if lower_is_better:
        best_vals = pivot.min(axis=1)
    else:
        best_vals = pivot.max(axis=1)
    pivot = pivot.loc[best_vals.sort_values(ascending=not lower_is_better).index]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot.index) * 0.4)))
    
    # Create grouped bar chart
    pivot.plot(kind='barh', ax=ax, width=0.8, colormap='viridis')
    
    metric_label = config.metric_labels.get(metric, metric.upper())
    ax.set_title(title or f'{metric_label} by Model and Fusion Method', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Model')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'method_comparison_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Method comparison saved to: {output_dir / f'method_comparison_{metric}.png'}")


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig):
    """Create heatmap comparing methods across models and strategies."""
    if metric not in df.columns or df.empty:
        print(f"Warning: No data for heatmap_{metric}")
        return
    
    # Create pivot table
    pivot = df.pivot_table(
        index='Method_Label',
        columns=['Model', 'Strategy_Label'],
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
    if metric in ['rmse', 'mae']:
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
    ax.set_title(f'{metric_label} Comparison Across Models & Strategies', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Model / Strategy')
    ax.set_ylabel('Fusion Method')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Heatmap saved to: {output_dir / f'heatmap_{metric}.png'}")


def create_summary_table(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Create summary table with mean and std for each model-strategy-method."""
    # Determine metrics
    tasks = df['Task'].unique()
    if len(tasks) == 1 and tasks[0] == 'regression':
        metrics = config.regression_metrics
    else:
        metrics = config.classification_metrics
    
    metrics = [m for m in metrics if m in df.columns]
    
    # Group and aggregate
    agg_dict = {}
    for metric in metrics:
        agg_dict[metric] = ['mean', 'std', 'count']
    
    summary = df.groupby(['Task', 'Model', 'Strategy_Label', 'Method_Label']).agg(agg_dict)
    summary = summary.round(4)
    
    # Save as CSV
    summary.to_csv(output_dir / 'summary_table.csv')
    print(f"✓ Summary table saved to: {output_dir / 'summary_table.csv'}")
    
    # Create flattened version
    flat_summary = summary.copy()
    flat_summary.columns = [f'{col[0]}_{col[1]}' for col in flat_summary.columns]
    flat_summary.reset_index().to_csv(output_dir / 'summary_table_flat.csv', index=False)
    
    return summary


# =======================================================================
#  MAIN FUNCTION
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flexible Experiment Results Aggregator'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=str, default='./results_summary',
                        help='Output directory for summary and figures')
    parser.add_argument('--patterns', nargs='+', 
                        default=['classification-bal-fusion-', 'classification-fusion-',
                                'regression-bal-fusion-', 'regression-fusion-'],
                        help='Folder patterns to match (e.g., classification-bal-fusion- regression-fusion-)')
    parser.add_argument('--fusion-subdir', type=str, default='leakage_safe_5fold',
                        help='Subdirectory under fusion_results to look for metrics')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'all'], 
                        default='all', help='Task type to aggregate')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to include')
    parser.add_argument('--strategies', nargs='+', default=['balanced', 'unbalanced', 'focal'],
                        help='Strategies to include')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific fusion methods to include')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to include in figures')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Discover experiments
    base_dir = Path(args.input_dir)
    experiments = discover_experiments(base_dir, args.patterns, args.fusion_subdir, config)
    
    if not experiments:
        print("\n" + "="*60)
        print("ERROR: No experiments found!")
        print("="*60)
        print(f"Base directory: {base_dir}")
        print(f"Patterns: {args.patterns}")
        print("\nExpected folder structure:")
        for pattern in args.patterns:
            print(f"  {pattern}<model_name>/")
        print(f"    └── fusion_results/{args.fusion_subdir}/")
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
    
    if args.strategies:
        df = df[df['Strategy'].isin(args.strategies)]
        main_df = main_df[main_df['Strategy'].isin(args.strategies)]
    
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
    print("Strategies found:", df['Strategy_Label'].unique().tolist())
    print("Methods found:", df['Method_Label'].unique().tolist())
    
    # Determine metrics to plot
    if args.metrics:
        metrics_to_plot = args.metrics
    else:
        tasks = df['Task'].unique()
        if len(tasks) == 1 and tasks[0] == 'regression':
            metrics_to_plot = ['rmse', 'mae', 'r2']
        else:
            metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'roc_auc', 'f1']
    
    # Save results
    main_df.to_csv(output_dir / 'main_results.csv', index=False)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate summary tables
    print(f"\n{'='*60}")
    print(f"GENERATING SUMMARY TABLES")
    print(f"{'='*60}")
    
    summary = create_summary_table(df, output_dir, config)
    
    # Generate figures
    print(f"\n{'='*60}")
    print(f"GENERATING FIGURES")
    print(f"{'='*60}")
    
    for metric in metrics_to_plot:
        if metric in df.columns and df[metric].notna().any():
            plot_heatmap(df, metric, output_dir, config)
            plot_method_comparison(df, metric, output_dir, config)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    ranking_metric = get_ranking_metric(df, config)
    metric_label = config.metric_labels.get(ranking_metric, ranking_metric.upper())
    
    print(f"\nBest by {metric_label}:")
    best_by_method = df.groupby('Method_Label')[ranking_metric].mean()
    
    if ranking_metric in ['rmse', 'mae']:
        best_by_method = best_by_method.sort_values(ascending=True)
    else:
        best_by_method = best_by_method.sort_values(ascending=False)
    
    for method, val in best_by_method.head(10).items():
        print(f"  {method}: {val:.4f}")
    
    print(f"\nBest by Model:")
    best_by_model = df.groupby('Model')[ranking_metric].mean()
    if ranking_metric in ['rmse', 'mae']:
        best_by_model = best_by_model.sort_values(ascending=True)
    else:
        best_by_model = best_by_model.sort_values(ascending=False)
    for model, val in best_by_model.items():
        print(f"  {model}: {val:.4f}")
    
    print(f"\nBest by Strategy:")
    best_by_strategy = df.groupby('Strategy_Label')[ranking_metric].mean()
    if ranking_metric in ['rmse', 'mae']:
        best_by_strategy = best_by_strategy.sort_values(ascending=True)
    else:
        best_by_strategy = best_by_strategy.sort_values(ascending=False)
    for strategy, val in best_by_strategy.items():
        print(f"  {strategy}: {val:.4f}")
    
    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()