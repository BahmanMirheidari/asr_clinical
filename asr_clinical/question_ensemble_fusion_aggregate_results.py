"""
Experiment Results Aggregator for Model-Specific Folders
Supports BOTH classification and regression experiments.

Folder Structure:
bal-fusion-<model_name>/           # Balanced models
fusion-<model_name>/               # Unbalanced models
regression-bal-fusion-<model_name>/ # Regression balanced
regression-fusion-<model_name>/     # Regression unbalanced

Inside each folder:
├── fusion_results/
│   ├── audio_only/
│   │   └── metrics.json
│   ├── text_only/
│   │   └── metrics.json
│   ├── early_fusion/
│   │   └── metrics.json
│   ├── late_fusion/
│   │   └── metrics.json
│   ├── model_based_fusion/
│   │   └── metrics.json
│   ├── confidence_weighted_fusion/
│   │   └── metrics.json
│   ├── interaction_stacking/
│   │   └── metrics.json
│   ├── mixture_of_experts/
│   │   └── metrics.json
│   └── mlp_early_fusion/
│       └── metrics.json
└── meta_test_metrics.json
"""

import os
import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
    
    # How to identify strategy from folder name
    strategy_prefixes: Dict[str, str] = field(default_factory=lambda: {
        'classification-bal-fusion-': 'balanced',
        'classification-fusion-': 'unbalanced',
        'regression-bal-fusion-': 'balanced',
        'regression-fusion-': 'unbalanced',
        'bal-fusion-': 'balanced',
        'fusion-': 'unbalanced',
        'focal-fusion-': 'focal'
    })
    
    # Task type (classification or regression)
    task: str = 'classification'  # Will be auto-detected
    
    # Fusion methods to look for
    fusion_methods: List[str] = field(default_factory=lambda: [
        'audio_only',
        'text_only',
        'early_fusion',
        'late_fusion',
        'model_based_fusion',
        'confidence_weighted_fusion',
        'interaction_stacking',
        'mixture_of_experts',
        'mlp_early_fusion'
    ])
    
    # Display names for fusion methods
    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        'audio_only': 'Audio-Only',
        'text_only': 'Text-Only',
        'early_fusion': 'Early Fusion',
        'late_fusion': 'Late Fusion',
        'model_based_fusion': 'Model-Based Stacking',
        'confidence_weighted_fusion': 'Confidence-Weighted',
        'interaction_stacking': 'Interaction Stacking',
        'mixture_of_experts': 'Mixture of Experts',
        'mlp_early_fusion': 'MLP Early Fusion'
    })
    
    # Strategy display names
    strategy_display_names: Dict[str, str] = field(default_factory=lambda: {
        'balanced': 'Balanced',
        'unbalanced': 'Unbalanced',
        'focal': 'Focal Loss'
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
    
    # Which metric to use for ranking (best)
    ranking_metric_classification: str = 'roc_auc'
    ranking_metric_regression: str = 'r2'  # Higher is better for R²


# =======================================================================
#  FOLDER PARSING FUNCTIONS
# =======================================================================

def detect_task_from_folder(folder_name: str) -> str:
    """Detect if folder is classification or regression."""
    if folder_name.startswith('classification-') or 'classification' in folder_name:
        return 'classification'
    elif folder_name.startswith('regression-') or 'regression' in folder_name:
        return 'regression'
    return 'classification'  # Default


def parse_folder_name(folder_name: str, config: ExperimentConfig) -> Dict[str, str]:
    """
    Parse folder name to extract task, strategy, and model.
    
    Examples:
        classification-bal-fusion-distilroberta-base → {'task': 'classification', 'strategy': 'balanced', 'model': 'distilroberta-base'}
        regression-fusion-deberta-v3 → {'task': 'regression', 'strategy': 'unbalanced', 'model': 'deberta-v3'}
    """
    result = {
        'task': None,
        'strategy': None,
        'model': None,
        'full_name': folder_name
    }
    
    # Detect task
    result['task'] = detect_task_from_folder(folder_name)
    
    # Check each prefix
    for prefix, strategy in config.strategy_prefixes.items():
        if folder_name.startswith(prefix):
            result['strategy'] = strategy
            # Remove prefix to get model name
            model_name = folder_name[len(prefix):]
            # Remove any trailing underscores or suffixes
            model_name = re.sub(r'[_\-].*$', '', model_name)
            result['model'] = model_name
            return result
    
    # If no prefix matches, try to infer
    if 'bal' in folder_name.lower():
        result['strategy'] = 'balanced'
    elif 'focal' in folder_name.lower():
        result['strategy'] = 'focal'
    else:
        result['strategy'] = 'unbalanced'
    
    # Extract model name
    clean_name = folder_name
    for prefix in config.strategy_prefixes.keys():
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
    
    # Remove any strategy suffixes
    clean_name = re.sub(r'[_\-](unbalanced|balanced|focal|no_balance|weighted).*$', '', clean_name)
    result['model'] = clean_name
    
    return result


# =======================================================================
#  EXPERIMENT DISCOVERY
# =======================================================================

def discover_experiments(base_dir: Path, config: ExperimentConfig) -> Dict:
    """
    Discover experiments by scanning folders matching naming conventions.
    """
    experiments = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n{'='*60}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*60}")
    print("\nLooking for folders matching patterns:")
    print("  - classification-bal-fusion-* (balanced classification)")
    print("  - classification-fusion-* (unbalanced classification)")
    print("  - regression-bal-fusion-* (balanced regression)")
    print("  - regression-fusion-* (unbalanced regression)")
    
    # Find all relevant folders
    all_folders = []
    
    # Look for classification folders
    all_folders.extend(list(base_dir.glob('classification-bal-fusion-*')))
    all_folders.extend([f for f in base_dir.glob('classification-fusion-*') 
                       if not f.name.startswith('classification-bal-')])
    
    # Look for regression folders
    all_folders.extend(list(base_dir.glob('regression-bal-fusion-*')))
    all_folders.extend([f for f in base_dir.glob('regression-fusion-*') 
                       if not f.name.startswith('regression-bal-')])
    
    # Also look for generic patterns
    all_folders.extend(list(base_dir.glob('bal-fusion-*')))
    all_folders.extend([f for f in base_dir.glob('fusion-*') 
                       if not f.name.startswith('bal-') and not f.name.startswith('focal-')])
    all_folders.extend(list(base_dir.glob('focal-fusion-*')))
    
    all_folders = list(set(all_folders))
    
    print(f"\nFound {len(all_folders)} potential experiment folders")
    
    # Process each folder
    for folder_path in all_folders:
        folder_name = folder_path.name
        
        # Parse folder name
        parsed = parse_folder_name(folder_name, config)
        model_name = parsed['model']
        strategy_key = parsed['strategy']
        task = parsed['task']
        
        # Create a composite key that includes task
        model_strategy_key = f"{task}_{model_name}"
        
        strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
        
        print(f"\nProcessing: {folder_name}")
        print(f"  Task: {task}")
        print(f"  Model: {model_name}")
        print(f"  Strategy: {strategy_key} ({strategy_display})")
        
        # Check if fusion_results directory exists
        fusion_dir = folder_path / 'fusion_results'
        if fusion_dir.exists():
            for method_dir in fusion_dir.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    
                    # Find method key
                    method_key = None
                    method_display = method_name
                    
                    if method_name in config.fusion_methods:
                        method_key = method_name
                        method_display = config.method_display_names.get(method_name, method_name)
                    else:
                        for known_method in config.fusion_methods:
                            if method_name == known_method or method_name.startswith(known_method):
                                method_key = known_method
                                method_display = config.method_display_names.get(known_method, method_name)
                                break
                    
                    if method_key is None:
                        method_lower = method_name.lower()
                        if 'audio' in method_lower:
                            method_key = 'audio_only'
                        elif 'text' in method_lower:
                            method_key = 'text_only'
                        elif 'early' in method_lower:
                            method_key = 'early_fusion'
                        elif 'late' in method_lower:
                            method_key = 'late_fusion'
                        elif 'stack' in method_lower or 'model_based' in method_lower:
                            method_key = 'model_based_fusion'
                        elif 'confidence' in method_lower or 'entropy' in method_lower:
                            method_key = 'confidence_weighted_fusion'
                        elif 'interaction' in method_lower:
                            method_key = 'interaction_stacking'
                        elif 'moe' in method_lower or 'expert' in method_lower:
                            method_key = 'mixture_of_experts'
                        elif 'mlp' in method_lower or 'neural' in method_lower:
                            method_key = 'mlp_early_fusion'
                        else:
                            method_key = method_name
                        method_display = config.method_display_names.get(method_key, method_name)
                    
                    # Find metrics file
                    metrics_file = method_dir / 'metrics.json'
                    if not metrics_file.exists():
                        possible_files = [
                            method_dir / 'fusion_metrics.json',
                            method_dir / 'test_metrics.json',
                            method_dir / 'cv_metrics.json'
                        ]
                        for pf in possible_files:
                            if pf.exists():
                                metrics_file = pf
                                break
                    
                    if metrics_file.exists():
                        experiments[model_strategy_key][strategy_key][method_key] = {
                            'task': task,
                            'method_display': method_display,
                            'strategy_display': strategy_display,
                            'dir': method_dir,
                            'metrics_file': metrics_file,
                            'predictions_file': method_dir / 'predictions.csv',
                            'confusion_matrix': next(iter(method_dir.glob('*confusion*.png')), None),
                            'roc_curve': next(iter(method_dir.glob('*roc*.png')), None)
                        }
                        print(f"    ✓ Found: {method_key} ({method_display})")
                    else:
                        print(f"    ⚠ No metrics found for: {method_name}")
        
        # Look for main results
        main_metrics = folder_path / 'meta_test_metrics.json'
        if main_metrics.exists():
            experiments[model_strategy_key][strategy_key]['main'] = {
                'task': task,
                'method_display': 'Main Results',
                'strategy_display': strategy_display,
                'dir': folder_path,
                'metrics_file': main_metrics,
                'predictions_file': folder_path / 'meta_test_predictions.csv',
                'confusion_matrix': None,
                'roc_curve': None
            }
            print(f"    ✓ Found main results")
    
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
#  DATA LOADING FUNCTIONS
# =======================================================================

def load_metrics(metrics_file: Path) -> Optional[Dict]:
    """Load metrics from JSON file."""
    if not metrics_file or not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except:
        return None


def extract_metrics_from_result(result: Dict, task: str) -> Dict:
    """Extract relevant metrics from result dictionary based on task."""
    extracted = {}
    
    if isinstance(result, dict):
        if 'accuracy' in result:
            extracted = result.copy()
        elif 'test_metrics' in result and isinstance(result['test_metrics'], dict):
            extracted = result['test_metrics'].copy()
        elif 'aggregate_metrics' in result and isinstance(result['aggregate_metrics'], dict):
            extracted = result['aggregate_metrics'].copy()
        elif 'cv_aggregate_metrics' in result and isinstance(result['cv_aggregate_metrics'], dict):
            extracted = result['cv_aggregate_metrics'].copy()
        elif 'cv_metrics' in result and isinstance(result['cv_metrics'], dict):
            extracted = result['cv_metrics'].copy()
    
    # Ensure all metrics are present
    if task == 'classification':
        standard_metrics = [
            'accuracy', 'sensitivity', 'specificity', 
            'precision', 'npv', 'f1', 'roc_auc',
            'macro_f1', 'balanced_accuracy'
        ]
    else:  # regression
        standard_metrics = [
            'rmse', 'mae', 'r2'
        ]
    
    for metric in standard_metrics:
        if metric not in extracted:
            extracted[metric] = None
    
    return extracted


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
                
                metrics = load_metrics(method_data.get('metrics_file'))
                if metrics is None:
                    continue
                
                extracted = extract_metrics_from_result(metrics, task)
                
                strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
                method_display = config.method_display_names.get(method_key, method_key)
                
                row = {
                    'Task': task,
                    'Model': model_name,
                    'Strategy': strategy_key,
                    'Strategy_Label': strategy_display,
                    'Method': method_key,
                    'Method_Label': method_display,
                    'Has_Probabilities': method_data.get('predictions_file', Path()).exists(),
                }
                
                # Add metrics based on task
                if task == 'classification':
                    metrics_to_add = config.classification_metrics
                else:
                    metrics_to_add = config.regression_metrics
                
                for metric in metrics_to_add:
                    row[metric] = extracted.get(metric)
                
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
            metrics = load_metrics(main_data.get('metrics_file'))
            if metrics is None:
                continue
            
            extracted = extract_metrics_from_result(metrics, task)
            
            strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
            
            row = {
                'Task': task,
                'Model': model_name,
                'Strategy': strategy_key,
                'Strategy_Label': strategy_display,
                'Best_K': extracted.get('avg_best_k', extracted.get('best_k', None)),
                'Selected_Questions': extracted.get('selected_questions', [])
            }
            
            if task == 'classification':
                for metric in config.classification_metrics:
                    row[metric] = extracted.get(metric)
            else:
                for metric in config.regression_metrics:
                    row[metric] = extracted.get(metric)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    all_metrics = config.classification_metrics + config.regression_metrics
    for metric in all_metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


def get_ranking_metric(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Get the appropriate ranking metric based on task."""
    tasks = df['Task'].unique()
    if len(tasks) == 0:
        return 'roc_auc'
    
    # If all rows are same task, use that task's ranking metric
    if len(tasks) == 1:
        task = tasks[0]
        if task == 'classification':
            return config.ranking_metric_classification
        else:
            return config.ranking_metric_regression
    
    # Mixed tasks - use a generic metric that exists in all rows
    # Check which metrics are available
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
        return 'rmse'  # Fallback


def get_best_per_config(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Get best performing method for each model-strategy combination."""
    best_rows = []
    ranking_metric = get_ranking_metric(df, config)
    
    for (task, model, strategy), group in df.groupby(['Task', 'Model', 'Strategy']):
        # Find best by ranking metric
        if ranking_metric in group.columns and group[ranking_metric].notna().any():
            # For RMSE, lower is better
            if ranking_metric == 'rmse':
                best_idx = group[ranking_metric].idxmin()
            else:
                best_idx = group[ranking_metric].idxmax()
            best_row = group.loc[best_idx].copy()
            best_row['Best_Method'] = best_row['Method_Label']
            best_rows.append(best_row)
        elif 'accuracy' in group.columns and group['accuracy'].notna().any():
            best_idx = group['accuracy'].idxmax()
            best_row = group.loc[best_idx].copy()
            best_row['Best_Method'] = best_row['Method_Label']
            best_rows.append(best_row)
    
    return pd.DataFrame(best_rows)


# =======================================================================
#  VISUALIZATION FUNCTIONS
# =======================================================================

def plot_heatmap_comparison(df: pd.DataFrame, metric: str, output_dir: Path,
                            config: ExperimentConfig, title: str = None):
    """Create heatmap comparing methods across models and strategies."""
    if metric not in df.columns:
        return
    
    # Create combined labels
    df['Model_Strategy'] = df['Model'] + '\n' + df['Strategy_Label']
    
    # Pivot data
    pivot = df.pivot_table(
        index='Method_Label',
        columns='Model_Strategy',
        values=metric,
        aggfunc='mean'
    )
    
    # Drop columns with all NaN
    pivot = pivot.dropna(axis=1, how='all')
    
    if pivot.empty:
        print(f"Warning: No data for heatmap_{metric}")
        return
    
    # Determine colormap based on metric
    if metric in ['rmse', 'mae']:
        cmap = 'RdYlGn'  # Lower is better for RMSE/MAE
    else:
        cmap = 'RdYlGn_r'  # Higher is better
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 
                                   max(8, len(pivot.index) * 0.6)))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                cbar_kws={'label': config.metric_labels.get(metric, metric.upper())},
                linewidths=0.5, linecolor='white',
                ax=ax, annot_kws={'fontsize': 8})
    
    metric_label = config.metric_labels.get(metric, metric.upper())
    ax.set_title(title or f'{metric_label} Comparison Across Models & Strategies', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Model / Strategy')
    ax.set_ylabel('Fusion Method')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Heatmap saved to: {output_dir / f'heatmap_{metric}.png'}")


def plot_bar_comparison(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig):
    """Create bar chart comparing methods for each model and strategy."""
    if metric not in df.columns:
        return
    
    # Get unique combinations
    tasks = df['Task'].unique()
    models = df['Model'].unique()
    strategies = df['Strategy_Label'].unique()
    
    if len(models) == 0 or len(strategies) == 0:
        print(f"Warning: No data for bar_{metric}")
        return
    
    # Determine if lower is better
    lower_is_better = metric in ['rmse', 'mae']
    
    fig, axes = plt.subplots(len(models), len(strategies),
                             figsize=(max(12, len(strategies) * 4),
                                     max(10, len(models) * 4)))
    
    if len(models) == 1:
        axes = [axes]
    if len(strategies) == 1:
        axes = [[ax] for ax in axes]
    
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            ax = axes[i][j] if len(models) > 1 else axes[j]
            
            subset = df[(df['Model'] == model) & 
                       (df['Strategy_Label'] == strategy)]
            
            if subset.empty:
                ax.set_title(f'{model}\n{strategy}\n(No data)')
                continue
            
            # Sort by metric
            subset = subset.sort_values(metric, ascending=lower_is_better)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(subset)))
            bars = ax.bar(subset['Method_Label'], subset[metric], color=colors)
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if not lower_is_better else 0),
                           f'{height:.3f}', ha='center', va='bottom',
                           fontsize=7, rotation=45)
            
            metric_label = config.metric_labels.get(metric, metric.upper())
            ax.set_title(f'{model}\n{strategy}', fontsize=10)
            ax.set_ylabel(metric_label)
            ax.set_xticklabels(subset['Method_Label'], rotation=45, ha='right', fontsize=8)
            
            # Set y-axis limits
            if lower_is_better:
                ax.set_ylim(0, max(subset[metric].max() * 1.2, 0.1))
            else:
                ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'bar_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Bar chart saved to: {output_dir / f'bar_{metric}.png'}")


def plot_best_method_heatmap(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Create heatmap showing best method for each model and strategy."""
    best_df = get_best_per_config(df, config)
    
    if best_df.empty:
        print("No data for best method heatmap")
        return
    
    # Pivot for heatmap
    pivot = best_df.pivot_table(
        index=['Task', 'Model'],
        columns='Strategy_Label',
        values='Best_Method',
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2),
                                   max(6, len(pivot.index) * 0.8)))
    
    sns.heatmap(pivot, annot=True, fmt='', cmap='coolwarm',
                cbar=False, linewidths=0.5, linecolor='white',
                ax=ax, annot_kws={'fontsize': 9})
    
    ax.set_title('Best Fusion Method per Configuration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Balance Strategy')
    ax.set_ylabel('Task / Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_method_heatmap.png', dpi=300)
    plt.close()
    print(f"✓ Best method heatmap saved to: {output_dir / 'best_method_heatmap.png'}")


def plot_radar_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig, top_n: int = 6):
    """Create radar chart comparing top methods across metrics."""
    ranking_metric = get_ranking_metric(df, config)
    
    # Get top methods by ranking metric
    if ranking_metric in df.columns:
        top_methods = df.groupby('Method_Label')[ranking_metric].mean().sort_values(
            ascending=(ranking_metric == 'rmse')  # Ascending for RMSE
        ).head(top_n).index.tolist()
    else:
        top_methods = df.groupby('Method_Label')['roc_auc'].mean().sort_values(
            ascending=False
        ).head(top_n).index.tolist()
    
    if not top_methods:
        print("No data for radar chart")
        return
    
    # Determine metrics based on task
    tasks = df['Task'].unique()
    if len(tasks) == 1 and tasks[0] == 'regression':
        metrics = ['rmse', 'mae', 'r2']
        labels = ['RMSE', 'MAE', 'R²']
        # For radar, we want higher = better, so invert RMSE and MAE
        invert = ['rmse', 'mae']
    else:
        metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'roc_auc']
        labels = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'AUC']
        invert = []
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_methods)))
    
    for idx, method in enumerate(top_methods):
        subset = df[df['Method_Label'] == method]
        values = []
        
        for metric in metrics:
            val = subset[metric].mean()
            if np.isnan(val):
                val = 0
            # Invert if lower is better
            if metric in invert:
                # Normalize to 0-1 range (assuming max RMSE ~1)
                val = max(0, 1 - val) if val > 0 else 0
            values.append(val)
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5,
                label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Top Fusion Methods Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300)
    plt.close()
    print(f"✓ Radar chart saved to: {output_dir / 'radar_comparison.png'}")


def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Create boxplot comparing balance strategies across methods."""
    ranking_metric = get_ranking_metric(df, config)
    
    if ranking_metric not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    data = []
    labels = []
    colors = []
    
    for method in df['Method_Label'].unique():
        subset = df[df['Method_Label'] == method]
        if subset.empty:
            continue
        
        for strategy in subset['Strategy_Label'].unique():
            vals = subset[subset['Strategy_Label'] == strategy][ranking_metric].dropna()
            if not vals.empty:
                data.extend(vals.tolist())
                labels.append(f'{method}\n{strategy}')
                is_balanced = 'Balanced' in strategy
                colors.append('blue' if is_balanced else 'red')
    
    if not data:
        print("No data for strategy comparison")
        return
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    metric_label = config.metric_labels.get(ranking_metric, ranking_metric.upper())
    ax.set_title(f'{metric_label} Distribution by Method and Balance Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits based on metric
    if ranking_metric in ['rmse', 'mae']:
        # For error metrics, lower is better
        pass
    else:
        ax.set_ylim(0, 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_boxplot.png', dpi=300)
    plt.close()
    print(f"✓ Strategy comparison saved to: {output_dir / 'strategy_comparison_boxplot.png'}")


def plot_model_comparison(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig):
    """Create grouped bar chart comparing models for each method and strategy."""
    if metric not in df.columns:
        return
    
    methods = df['Method_Label'].unique()
    strategies = df['Strategy_Label'].unique()
    
    if len(methods) == 0 or len(strategies) == 0:
        print(f"Warning: No data for model_comparison_{metric}")
        return
    
    lower_is_better = metric in ['rmse', 'mae']
    
    fig, axes = plt.subplots(len(methods), len(strategies),
                             figsize=(max(14, len(strategies) * 4),
                                     max(12, len(methods) * 4)))
    
    if len(methods) == 1:
        axes = [axes]
    if len(strategies) == 1:
        axes = [[ax] for ax in axes]
    
    for i, method in enumerate(methods):
        for j, strategy in enumerate(strategies):
            ax = axes[i][j] if len(methods) > 1 else axes[j]
            
            subset = df[(df['Method_Label'] == method) & 
                       (df['Strategy_Label'] == strategy)]
            
            if subset.empty:
                ax.set_title(f'{method}\n{strategy}\n(No data)')
                continue
            
            # Group by model
            model_means = subset.groupby('Model')[metric].mean()
            if lower_is_better:
                model_means = model_means.sort_values(ascending=True)
            else:
                model_means = model_means.sort_values(ascending=False)
            
            if model_means.empty:
                continue
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_means)))
            bars = ax.barh(model_means.index, model_means.values, color=colors)
            
            for bar in bars:
                width = bar.get_width()
                if not np.isnan(width):
                    ax.text(width + (0.01 if not lower_is_better else 0), 
                           bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', va='center', fontsize=9)
            
            metric_label = config.metric_labels.get(metric, metric.upper())
            ax.set_title(f'{method}\n{strategy}', fontsize=10)
            ax.set_xlabel(metric_label)
            if not lower_is_better:
                ax.set_xlim(0, max(1, model_means.max() * 1.1))
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'model_comparison_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Model comparison saved to: {output_dir / f'model_comparison_{metric}.png'}")


def create_summary_table(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Create summary table with mean and std for each model-strategy-method."""
    # Determine which metrics to include based on task
    tasks = df['Task'].unique()
    
    if len(tasks) == 1 and tasks[0] == 'regression':
        metrics = config.regression_metrics
    else:
        metrics = config.classification_metrics + ['r2']  # Include R² if available
    
    # Filter to metrics that exist in the data
    metrics = [m for m in metrics if m in df.columns]
    
    # Group by task, model, strategy, method
    agg_dict = {}
    for metric in metrics:
        agg_dict[metric] = ['mean', 'std', 'count']
    
    summary = df.groupby(['Task', 'Model', 'Strategy_Label', 'Method_Label']).agg(agg_dict).round(4)
    
    # Save as CSV
    summary.to_csv(output_dir / 'summary_table.csv')
    print(f"✓ Summary table saved to: {output_dir / 'summary_table.csv'}")
    
    # Create flattened version
    flat_summary = summary.copy()
    flat_summary.columns = [f'{col[0]}_{col[1]}' for col in flat_summary.columns]
    flat_summary.reset_index().to_csv(output_dir / 'summary_table_flat.csv', index=False)
    
    return summary


def perform_statistical_tests(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig):
    """Perform statistical tests comparing methods."""
    ranking_metric = get_ranking_metric(df, config)
    
    if ranking_metric not in df.columns:
        return
    
    results = []
    methods = df['Method_Label'].unique()
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            vals1 = df[df['Method_Label'] == method1][ranking_metric].dropna()
            vals2 = df[df['Method_Label'] == method2][ranking_metric].dropna()
            
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            
            try:
                stat, p_value = stats.wilcoxon(vals1, vals2)
                # Determine which is better
                if ranking_metric in ['rmse', 'mae']:
                    better = 'method1' if vals1.mean() < vals2.mean() else 'method2'
                else:
                    better = 'method1' if vals1.mean() > vals2.mean() else 'method2'
                
                results.append({
                    'Method_1': method1,
                    'Method_2': method2,
                    'Metric': ranking_metric,
                    'Mean_1': vals1.mean(),
                    'Mean_2': vals2.mean(),
                    'Better': better,
                    'p_value': p_value,
                    'statistic': stat,
                    'significant': p_value < 0.05
                })
            except:
                continue
    
    if results:
        stat_df = pd.DataFrame(results)
        stat_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
        print(f"✓ Statistical tests saved to: {output_dir / 'statistical_tests.csv'}")


# =======================================================================
#  MAIN FUNCTION
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment Results Aggregator (Classification + Regression)'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=str, default='./results_summary',
                        help='Output directory for summary and figures')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'all'], 
                        default='all', help='Task type to aggregate')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to include')
    parser.add_argument('--strategies', nargs='+', default=['balanced', 'unbalanced'],
                        help='Strategies to include')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific fusion methods to include')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to include in figures')
    parser.add_argument('--top-n', type=int, default=6,
                        help='Number of top methods to show in radar chart')
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
    experiments = discover_experiments(base_dir, config)
    
    if not experiments:
        print("\n" + "="*60)
        print("ERROR: No experiments found!")
        print("="*60)
        print(f"Base directory: {base_dir}")
        print("\nExpected folder structure:")
        print("  classification-bal-fusion-<model_name>/")
        print("  classification-fusion-<model_name>/")
        print("  regression-bal-fusion-<model_name>/")
        print("  regression-fusion-<model_name>/")
        return
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    df = aggregate_experiment_results(experiments, config)
    main_df = aggregate_main_results(experiments, config)
    
    # Filter by task
    if args.task != 'all':
        df = df[df['Task'] == args.task]
        main_df = main_df[main_df['Task'] == args.task]
    
    # Filter by models if specified
    if args.models:
        df = df[df['Model'].isin(args.models)]
        main_df = main_df[main_df['Model'].isin(args.models)]
    
    # Filter by strategies if specified
    if args.strategies:
        df = df[df['Strategy'].isin(args.strategies)]
        main_df = main_df[main_df['Strategy'].isin(args.strategies)]
    
    # Filter by methods if specified
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
            metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'roc_auc']
    
    # Generate summary tables
    print(f"\n{'='*60}")
    print(f"GENERATING SUMMARY TABLES")
    print(f"{'='*60}")
    
    summary = create_summary_table(df, output_dir, config)
    
    # Save results
    main_df.to_csv(output_dir / 'main_results.csv', index=False)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate figures
    print(f"\n{'='*60}")
    print(f"GENERATING FIGURES")
    print(f"{'='*60}")
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plot_heatmap_comparison(df, metric, output_dir, config)
            plot_bar_comparison(df, metric, output_dir, config)
            plot_model_comparison(df, metric, output_dir, config)
    
    plot_best_method_heatmap(df, output_dir, config)
    plot_radar_comparison(df, output_dir, config, args.top_n)
    plot_strategy_comparison(df, output_dir, config)
    
    perform_statistical_tests(df, output_dir, config)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Get ranking metric
    ranking_metric = get_ranking_metric(df, config)
    metric_label = config.metric_labels.get(ranking_metric, ranking_metric.upper())
    
    print(f"\nBest by {metric_label}:")
    best_by_method = df.groupby('Method_Label')[ranking_metric].mean()
    
    # For RMSE/MAE, lower is better
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
'''
For Classification Only:
bash
python aggregate_results.py \
    --input-dir /path/to/your/outputs \
    --output-dir ./classification_results \
    --task classification \
    --models distilroberta-base microsoft/MiniLM-L12-H384-uncased microsoft/deberta-v3-base \
    --strategies balanced unbalanced \
    --metrics accuracy sensitivity specificity roc_auc macro_f1 \
    --top-n 6
For Regression Only:
bash
python aggregate_results.py \
    --input-dir /path/to/your/outputs \
    --output-dir ./regression_results \
    --task regression \
    --models distilroberta-base microsoft/MiniLM-L12-H384-uncased \
    --strategies balanced unbalanced \
    --metrics rmse mae r2 \
    --top-n 6
For Both:
bash
python aggregate_results.py \
    --input-dir /path/to/your/outputs \
    --output-dir ./all_results \
    --task all \
    --models distilroberta-base microsoft/MiniLM-L12-H384-uncased microsoft/deberta-v3-base \
    --strategies balanced unbalanced \
    --top-n 6
Expected Output for Regression
text
📊 SUMMARY STATISTICS
======================================================================

Best by R²:
  Mixture of Experts: 0.892
  Interaction Stacking: 0.875
  Model-Based Stacking: 0.861
  MLP Early Fusion: 0.854
  Late Fusion: 0.843

Best by Model:
  microsoft/deberta-v3-base: 0.878
  microsoft/MiniLM-L12-H384-uncased: 0.862
  distilroberta-base: 0.845

Best by Strategy:
  Balanced: 0.875
  Unbalanced: 0.842

'''