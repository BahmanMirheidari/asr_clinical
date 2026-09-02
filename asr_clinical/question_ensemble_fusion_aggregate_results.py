"""
Experiment Results Aggregator for Model-Specific Folders

Folder Structure:
bal-fusion-<model_name>/           # Balanced models
fusion-<model_name>/               # Unbalanced models

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

Generates summary tables and publication-ready figures.
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
        'bal-fusion-': 'balanced',
        'fusion-': 'unbalanced',
        'focal-fusion-': 'focal'
    })
    
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
        'balanced': 'Balanced (CE)',
        'unbalanced': 'Unbalanced (CE)',
        'focal': 'Focal Loss (γ=2.0)'
    })
    
    # Metrics to extract
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'sensitivity', 'specificity', 
        'precision', 'npv', 'f1', 'roc_auc',
        'macro_f1', 'balanced_accuracy'
    ])
    
    metric_labels: Dict[str, str] = field(default_factory=lambda: {
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'precision': 'PPV',
        'npv': 'NPV',
        'f1': 'F1 Score',
        'roc_auc': 'AUC-ROC',
        'macro_f1': 'Macro F1',
        'balanced_accuracy': 'Balanced Accuracy'
    })


# =======================================================================
#  FOLDER PARSING FUNCTIONS
# =======================================================================

def parse_folder_name(folder_name: str, config: ExperimentConfig) -> Dict[str, str]:
    """
    Parse folder name to extract strategy and model.
    
    Examples:
        bal-fusion-distilroberta-base → {'strategy': 'balanced', 'model': 'distilroberta-base'}
        fusion-deberta-v3 → {'strategy': 'unbalanced', 'model': 'deberta-v3'}
        focal-fusion-MiniLM → {'strategy': 'focal', 'model': 'MiniLM'}
    
    Returns:
        {'strategy': 'balanced', 'model': 'distilroberta-base'}
    """
    result = {
        'strategy': None,
        'model': None,
        'full_name': folder_name
    }
    
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
    
    # If no prefix matches, try to infer from folder name
    # Check if it contains 'bal' or 'fusion'
    if 'bal' in folder_name.lower():
        result['strategy'] = 'balanced'
    elif 'focal' in folder_name.lower():
        result['strategy'] = 'focal'
    else:
        result['strategy'] = 'unbalanced'
    
    # Extract model name
    # Remove common prefixes
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
    
    Returns:
        Structure: {
            model_name: {
                strategy_key: {
                    method_key: {
                        'method_display': str,
                        'strategy_display': str,
                        'dir': Path,
                        'metrics_file': Path,
                        'predictions_file': Path,
                        'confusion_matrix': Path,
                        'roc_curve': Path
                    }
                }
            }
        }
    """
    experiments = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n{'='*60}")
    print(f"DISCOVERING EXPERIMENTS IN: {base_dir}")
    print(f"{'='*60}")
    print("\nLooking for folders matching patterns:")
    print("  - bal-fusion-* (balanced)")
    print("  - fusion-* (unbalanced)")
    print("  - focal-fusion-* (focal)")
    
    # Find all relevant folders
    all_folders = []
    
    # Look for bal-fusion-* folders
    bal_folders = list(base_dir.glob('bal-fusion-*'))
    all_folders.extend(bal_folders)
    
    # Look for fusion-* folders (excluding bal-fusion and focal-fusion)
    fusion_folders = [f for f in base_dir.glob('fusion-*') 
                     if not f.name.startswith('bal-') and not f.name.startswith('focal-')]
    all_folders.extend(fusion_folders)
    
    # Look for focal-fusion-* folders
    focal_folders = list(base_dir.glob('focal-fusion-*'))
    all_folders.extend(focal_folders)
    
    # Also look for any subdirectories that might contain fusion results
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and subdir.name not in [f.name for f in all_folders]:
            # Check if it contains fusion_results
            if (subdir / 'fusion_results').exists():
                all_folders.append(subdir)
    
    all_folders = list(set(all_folders))
    
    print(f"\nFound {len(all_folders)} potential experiment folders")
    
    # Process each folder
    for folder_path in all_folders:
        folder_name = folder_path.name
        
        # Parse folder name
        parsed = parse_folder_name(folder_name, config)
        model_name = parsed['model']
        strategy_key = parsed['strategy']
        strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
        
        print(f"\nProcessing: {folder_name}")
        print(f"  Model: {model_name}")
        print(f"  Strategy: {strategy_key} ({strategy_display})")
        
        # Check if fusion_results directory exists
        fusion_dir = folder_path / 'fusion_results'
        if fusion_dir.exists():
            # Look for method subdirectories
            for method_dir in fusion_dir.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    
                    # Check if this is a known method
                    method_key = None
                    method_display = method_name
                    
                    # Try exact match first
                    if method_name in config.fusion_methods:
                        method_key = method_name
                        method_display = config.method_display_names.get(method_name, method_name)
                    else:
                        # Try partial match
                        for known_method in config.fusion_methods:
                            if method_name == known_method or method_name.startswith(known_method):
                                method_key = known_method
                                method_display = config.method_display_names.get(known_method, method_name)
                                break
                    
                    # If still not found, try to infer from name
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
                            # Use as is
                            method_key = method_name
                        
                        method_display = config.method_display_names.get(method_key, method_name)
                    
                    # Find metrics file
                    metrics_file = method_dir / 'metrics.json'
                    if not metrics_file.exists():
                        # Try other possible names
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
                        experiments[model_name][strategy_key][method_key] = {
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
        
        # Also look for main results (meta_test_metrics.json)
        main_metrics = folder_path / 'meta_test_metrics.json'
        if main_metrics.exists():
            # Add as a special entry
            experiments[model_name][strategy_key]['main'] = {
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
    for model, strategies in experiments.items():
        print(f"\nModel: {model}")
        for strategy, methods in strategies.items():
            strat_display = config.strategy_display_names.get(strategy, strategy)
            method_count = len([m for m in methods if m != 'main'])
            print(f"  {strat_display}: {method_count} methods")
            total_methods += method_count
            for method in methods:
                if method != 'main':
                    print(f"    - {method}")
    
    print(f"\nTotal: {len(experiments)} models, {total_methods} fusion method results")
    
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


def extract_metrics_from_result(result: Dict) -> Dict:
    """Extract relevant metrics from result dictionary."""
    extracted = {}
    
    if isinstance(result, dict):
        # Try different nested structures
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
    standard_metrics = [
        'accuracy', 'sensitivity', 'specificity', 
        'precision', 'npv', 'f1', 'roc_auc',
        'macro_f1', 'balanced_accuracy'
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
    
    for model_name, model_data in experiments.items():
        for strategy_key, strategy_data in model_data.items():
            for method_key, method_data in strategy_data.items():
                # Skip main results for now (we'll handle separately)
                if method_key == 'main':
                    continue
                
                # Load metrics
                metrics = load_metrics(method_data.get('metrics_file'))
                if metrics is None:
                    continue
                
                extracted = extract_metrics_from_result(metrics)
                
                # Get display names
                strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
                method_display = config.method_display_names.get(method_key, method_key)
                
                row = {
                    'Model': model_name,
                    'Strategy': strategy_key,
                    'Strategy_Label': strategy_display,
                    'Method': method_key,
                    'Method_Label': method_display,
                    'Has_Probabilities': method_data.get('predictions_file', Path()).exists(),
                }
                
                # Add metrics
                for metric in config.metrics:
                    row[metric] = extracted.get(metric)
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert metrics to numeric
    for metric in config.metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


def aggregate_main_results(experiments: Dict, config: ExperimentConfig) -> pd.DataFrame:
    """Aggregate main results (meta_test_metrics.json) for each run."""
    rows = []
    
    for model_name, model_data in experiments.items():
        for strategy_key, strategy_data in model_data.items():
            if 'main' not in strategy_data:
                continue
            
            main_data = strategy_data['main']
            metrics = load_metrics(main_data.get('metrics_file'))
            if metrics is None:
                continue
            
            extracted = extract_metrics_from_result(metrics)
            
            strategy_display = config.strategy_display_names.get(strategy_key, strategy_key)
            
            row = {
                'Model': model_name,
                'Strategy': strategy_key,
                'Strategy_Label': strategy_display,
                'Best_K': extracted.get('avg_best_k', extracted.get('best_k', None)),
                'Selected_Questions': extracted.get('selected_questions', [])
            }
            
            for metric in config.metrics:
                row[metric] = extracted.get(metric)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    for metric in config.metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    return df


def get_best_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Get best performing method for each model-strategy combination."""
    best_rows = []
    
    for (model, strategy), group in df.groupby(['Model', 'Strategy']):
        # Find best by AUC
        if group['roc_auc'].notna().any():
            best_idx = group['roc_auc'].idxmax()
            best_row = group.loc[best_idx].copy()
            best_row['Best_Method'] = best_row['Method_Label']
            best_rows.append(best_row)
        elif group['accuracy'].notna().any():
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
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.8), 
                                   max(8, len(pivot.index) * 0.6)))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
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
    models = df['Model'].unique()
    strategies = df['Strategy_Label'].unique()
    
    if len(models) == 0 or len(strategies) == 0:
        print(f"Warning: No data for bar_{metric}")
        return
    
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
            
            subset = subset.sort_values(metric, ascending=False)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(subset)))
            bars = ax.bar(subset['Method_Label'], subset[metric], color=colors)
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom',
                           fontsize=7, rotation=45)
            
            metric_label = config.metric_labels.get(metric, metric.upper())
            ax.set_title(f'{model}\n{strategy}', fontsize=10)
            ax.set_ylabel(metric_label)
            ax.set_xticklabels(subset['Method_Label'], rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'bar_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Bar chart saved to: {output_dir / f'bar_{metric}.png'}")


def plot_best_method_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing best method for each model and strategy."""
    best_df = get_best_per_config(df)
    
    if best_df.empty:
        print("No data for best method heatmap")
        return
    
    # Pivot for heatmap
    pivot = best_df.pivot_table(
        index='Model',
        columns='Strategy_Label',
        values='Best_Method',
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2),
                                   max(6, len(pivot.index) * 0.8)))
    
    sns.heatmap(pivot, annot=True, fmt='', cmap='coolwarm',
                cbar=False, linewidths=0.5, linecolor='white',
                ax=ax, annot_kws={'fontsize': 10})
    
    ax.set_title('Best Fusion Method per Configuration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Balance Strategy')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_method_heatmap.png', dpi=300)
    plt.close()
    print(f"✓ Best method heatmap saved to: {output_dir / 'best_method_heatmap.png'}")


def plot_radar_comparison(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig, top_n: int = 6):
    """Create radar chart comparing top methods across metrics."""
    # Get top methods by AUC
    top_methods = df.groupby('Method_Label')['roc_auc'].mean().sort_values(
        ascending=False
    ).head(top_n).index.tolist()
    
    if not top_methods:
        print("No data for radar chart")
        return
    
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'npv', 'roc_auc']
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC']
    
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


def plot_strategy_comparison(df: pd.DataFrame, output_dir: Path):
    """Create boxplot comparing balance strategies across methods."""
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
            vals = subset[subset['Strategy_Label'] == strategy]['roc_auc'].dropna()
            if not vals.empty:
                data.extend(vals.tolist())
                labels.append(f'{method}\n{strategy}')
                is_balanced = 'Balanced' in strategy or 'Focal' in strategy
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
    
    ax.set_title('AUC Distribution by Method and Balance Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_boxplot.png', dpi=300)
    plt.close()
    print(f"✓ Strategy comparison saved to: {output_dir / 'strategy_comparison_boxplot.png'}")


def plot_model_comparison(df: pd.DataFrame, metric: str, output_dir: Path, config: ExperimentConfig):
    """Create grouped bar chart comparing models for each method and strategy."""
    methods = df['Method_Label'].unique()
    strategies = df['Strategy_Label'].unique()
    
    if len(methods) == 0 or len(strategies) == 0:
        print(f"Warning: No data for model_comparison_{metric}")
        return
    
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
            model_means = subset.groupby('Model')[metric].mean().sort_values(ascending=True)
            
            if model_means.empty:
                continue
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_means)))
            bars = ax.barh(model_means.index, model_means.values, color=colors)
            
            for bar in bars:
                width = bar.get_width()
                if not np.isnan(width):
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', va='center', fontsize=9)
            
            metric_label = config.metric_labels.get(metric, metric.upper())
            ax.set_title(f'{method}\n{strategy}', fontsize=10)
            ax.set_xlabel(metric_label)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'model_comparison_{metric}.png', dpi=300)
    plt.close()
    print(f"✓ Model comparison saved to: {output_dir / f'model_comparison_{metric}.png'}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create summary table with mean and std for each model-strategy-method."""
    summary = df.groupby(['Model', 'Strategy_Label', 'Method_Label']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'sensitivity': ['mean', 'std', 'count'],
        'specificity': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'npv': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std', 'count']
    }).round(4)
    
    # Save as CSV
    summary.to_csv(output_dir / 'summary_table.csv')
    print(f"✓ Summary table saved to: {output_dir / 'summary_table.csv'}")
    
    # Create flattened version
    flat_summary = summary.copy()
    flat_summary.columns = [f'{col[0]}_{col[1]}' for col in flat_summary.columns]
    flat_summary.reset_index().to_csv(output_dir / 'summary_table_flat.csv', index=False)
    
    return summary


def create_latex_table(summary_df: pd.DataFrame, output_dir: Path):
    """Create LaTeX table from summary data."""
    # Select best method per model-strategy
    best_auc = summary_df.groupby(['Model', 'Strategy_Label'])['roc_auc_mean'].idxmax()
    best_rows = summary_df.loc[best_auc].reset_index()
    
    latex_lines = []
    latex_lines.append('\\begin{table}[htbp]')
    latex_lines.append('\\centering')
    latex_lines.append('\\caption{Best Fusion Method per Configuration}')
    latex_lines.append('\\label{tab:best_methods}')
    latex_lines.append('\\begin{tabular}{llrrrrrr}')
    latex_lines.append('\\hline')
    latex_lines.append('Model & Strategy & Method & Accuracy & Sensitivity & Specificity & AUC \\\\')
    latex_lines.append('\\hline')
    
    for _, row in best_rows.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['Strategy_Label']} & {row['Method_Label']} & "
            f"{row['accuracy_mean']:.3f} & {row['sensitivity_mean']:.3f} & "
            f"{row['specificity_mean']:.3f} & {row['roc_auc_mean']:.3f} \\\\"
        )
    
    latex_lines.append('\\hline')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')
    
    with open(output_dir / 'best_methods_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ LaTeX table saved to: {output_dir / 'best_methods_table.tex'}")


def perform_statistical_tests(df: pd.DataFrame, output_dir: Path):
    """Perform statistical tests comparing methods."""
    results = []
    
    methods = df['Method_Label'].unique()
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            auc1 = df[df['Method_Label'] == method1]['roc_auc'].dropna()
            auc2 = df[df['Method_Label'] == method2]['roc_auc'].dropna()
            
            if len(auc1) < 2 or len(auc2) < 2:
                continue
            
            try:
                stat, p_value = stats.wilcoxon(auc1, auc2)
                results.append({
                    'Method_1': method1,
                    'Method_2': method2,
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
    
    return results


# =======================================================================
#  MAIN FUNCTION
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experiment Results Aggregator for Model-Specific Folders'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Base directory containing experiment folders')
    parser.add_argument('--output-dir', type=str, default='./results_summary',
                        help='Output directory for summary and figures')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to include (e.g., distilroberta-base deberta-v3)')
    parser.add_argument('--strategies', nargs='+', default=['balanced', 'unbalanced'],
                        help='Strategies to include (balanced, unbalanced, focal)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific fusion methods to include')
    parser.add_argument('--metrics', nargs='+',
                        default=['accuracy', 'sensitivity', 'specificity', 'roc_auc'],
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
        print("  bal-fusion-<model_name>/")
        print("  fusion-<model_name>/")
        print("\nInside each folder:")
        print("  fusion_results/<method>/metrics.json")
        return
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    df = aggregate_experiment_results(experiments, config)
    main_df = aggregate_main_results(experiments, config)
    
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
        print("Try adjusting your filters or check the directory structure.")
        return
    
    # Print summary of discovered data
    print("\nModels found:", df['Model'].unique().tolist())
    print("Strategies found:", df['Strategy_Label'].unique().tolist())
    print("Methods found:", df['Method_Label'].unique().tolist())
    
    # Generate summary tables
    print(f"\n{'='*60}")
    print(f"GENERATING SUMMARY TABLES")
    print(f"{'='*60}")
    
    summary = create_summary_table(df, output_dir)
    create_latex_table(summary, output_dir)
    
    # Save main results
    main_df.to_csv(output_dir / 'main_results.csv', index=False)
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate figures
    print(f"\n{'='*60}")
    print(f"GENERATING FIGURES")
    print(f"{'='*60}")
    
    # 1. Heatmap for each metric
    for metric in args.metrics:
        plot_heatmap_comparison(df, metric, output_dir, config)
    
    # 2. Bar charts
    for metric in args.metrics:
        plot_bar_comparison(df, metric, output_dir, config)
    
    # 3. Model comparison
    for metric in args.metrics:
        plot_model_comparison(df, metric, output_dir, config)
    
    # 4. Best method heatmap
    plot_best_method_heatmap(df, output_dir)
    
    # 5. Radar chart
    plot_radar_comparison(df, output_dir, config, args.top_n)
    
    # 6. Strategy comparison
    plot_strategy_comparison(df, output_dir)
    
    # Statistical tests
    perform_statistical_tests(df, output_dir)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nOverall best AUC by method:")
    best_by_method = df.groupby('Method_Label')['roc_auc'].mean().sort_values(ascending=False)
    for method, auc in best_by_method.head(10).items():
        print(f"  {method}: {auc:.4f}")
    
    print(f"\nOverall best AUC by model:")
    best_by_model = df.groupby('Model')['roc_auc'].mean().sort_values(ascending=False)
    for model, auc in best_by_model.items():
        print(f"  {model}: {auc:.4f}")
    
    print(f"\nOverall best AUC by strategy:")
    best_by_strategy = df.groupby('Strategy_Label')['roc_auc'].mean().sort_values(ascending=False)
    for strategy, auc in best_by_strategy.items():
        print(f"  {strategy}: {auc:.4f}")
    
    # Print the discovered structure
    print(f"\n{'='*60}")
    print(f"DISCOVERED STRUCTURE")
    print(f"{'='*60}")
    for model, strategies in experiments.items():
        print(f"\n{model}:")
        for strategy, methods in strategies.items():
            strat_display = config.strategy_display_names.get(strategy, strategy)
            method_count = len([m for m in methods if m != 'main'])
            print(f"  {strat_display}: {method_count} methods")
            for method in methods:
                if method != 'main':
                    print(f"    - {method}")
    
    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*60}")
    
    print("\nTo include these figures in your paper:")
    print("  1. Copy figures from the output directory")
    print("  2. Use summary_table.csv for numeric results")
    print("  3. Use best_methods_table.tex for LaTeX table")
    print("  4. Use statistical_tests.csv for significance testing")


if __name__ == "__main__":
    main()

'''
How to Use This Aggregator
1. Run the Aggregator:
bash
python aggregate_results.py \
    --input-dir /path/to/your/experiment/outputs \
    --output-dir ./paper_results \
    --models distilroberta-base microsoft/MiniLM-L12-H384-uncased microsoft/deberta-v3-base \
    --strategies balanced unbalanced \
    --metrics accuracy sensitivity specificity roc_auc \
    --top-n 8
2. Your Folder Structure:
text
/path/to/your/experiment/outputs/
├── bal-fusion-distilroberta-base/
│   ├── fusion_results/
│   │   ├── audio_only/
│   │   │   └── metrics.json
│   │   ├── text_only/
│   │   │   └── metrics.json
│   │   ├── early_fusion/
│   │   │   └── metrics.json
│   │   ├── late_fusion/
│   │   │   └── metrics.json
│   │   ├── model_based_fusion/
│   │   │   └── metrics.json
│   │   ├── confidence_weighted_fusion/
│   │   │   └── metrics.json
│   │   ├── interaction_stacking/
│   │   │   └── metrics.json
│   │   ├── mixture_of_experts/
│   │   │   └── metrics.json
│   │   └── mlp_early_fusion/
│   │       └── metrics.json
│   └── meta_test_metrics.json
├── fusion-distilroberta-base/
│   └── (same structure)
├── bal-fusion-MiniLM/
│   └── (same structure)
├── fusion-MiniLM/
│   └── (same structure)
├── bal-fusion-deberta-v3/
│   └── (same structure)
└── fusion-deberta-v3/
    └── (same structure)
3. Output Files:
text
paper_results/
├── all_results.csv                    # All aggregated results
├── main_results.csv                   # Main results (meta_test_metrics.json)
├── summary_table.csv                  # Mean/std summary
├── summary_table_flat.csv             # Flattened summary table
├── best_methods_table.tex             # LaTeX table for paper
├── statistical_tests.csv              # Statistical significance tests
│
├── heatmap_accuracy.png               # Heatmap comparisons
├── heatmap_sensitivity.png
├── heatmap_specificity.png
├── heatmap_roc_auc.png
│
├── bar_accuracy.png                   # Bar charts
├── bar_sensitivity.png
├── bar_specificity.png
├── bar_roc_auc.png
│
├── model_comparison_accuracy.png      # Model comparison
├── model_comparison_sensitivity.png
├── model_comparison_specificity.png
├── model_comparison_roc_auc.png
│
├── best_method_heatmap.png            # Best method per config
├── radar_comparison.png               # Radar chart
└── strategy_comparison_boxplot.png    # Strategy comparison
4. Example Command with All Models:
bash
python aggregate_results.py \
    --input-dir /mnt/parscratch/users/ac1bm/MND-expr/outputs \
    --output-dir ./paper_results \
    --models distilroberta-base microsoft/MiniLM-L12-H384-uncased microsoft/deberta-v3-base \
    --strategies balanced unbalanced \
    --metrics accuracy sensitivity specificity roc_auc \
    --top-n 8 \
    --verbose
5. What Each Figure Shows:
Figure  Purpose
Heatmap Compare all methods across models and strategies for each metric
Bar Chart   Compare methods within each model-strategy combination
Model Comparison    Compare models for each method-strategy combination
Best Method Heatmap Shows best method for each configuration
Radar Chart Multi-metric comparison of top methods
Boxplot Distribution of AUC across strategies
6. Example Output Summary:
text
======================================================================
SUMMARY STATISTICS
======================================================================

Overall best AUC by method:
  Mixture of Experts: 0.934
  Interaction Stacking: 0.921
  Model-Based Stacking: 0.918
  Late Fusion: 0.889
  Early Fusion: 0.872
  Confidence-Weighted: 0.902
  MLP Early Fusion: 0.910
  Text-Only: 0.842
  Audio-Only: 0.783

Overall best AUC by model:
  microsoft/deberta-v3-base: 0.905
  microsoft/MiniLM-L12-H384-uncased: 0.895
  distilroberta-base: 0.878

Overall best AUC by strategy:
  Focal Loss (γ=2.0): 0.912
  Balanced (CE): 0.895
  Unbalanced (CE): 0.862
This aggregator will automatically discover and process all your bal-fusion-* and fusion-* folders!


'''