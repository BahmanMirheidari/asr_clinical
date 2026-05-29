# select_best_model.py
#!/usr/bin/env python3
"""
Comprehensive model selection tool for comparing trained models across different
transformer architectures (RoBERTa, DistilBERT, BERT, etc.)

Usage:
    python select_best_model.py --results-dir /path/to/results --task classification
    python select_best_model.py --results-dir /path/to/results --task regression --metric rmse
    python select_best_model.py --results-dir /path/to/results --compare-all --statistical-test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, mean_squared_error, r2_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Loading Functions
# ============================================================================

def find_model_dirs(results_dir: Path, pattern: str = "*") -> List[Path]:
    """Find all model output directories"""
    model_dirs = [d for d in results_dir.glob(pattern) if d.is_dir() and (d / "meta_model.joblib").exists()]
    return model_dirs


def load_model_results(model_dir: Path) -> Dict:
    """Load all results from a model directory"""
    results = {
        "model_name": model_dir.name,
        "model_path": str(model_dir),
        "has_test": False,
        "has_cv": False,
        "is_ensemble": False,
        "task": "unknown",
        "metrics": {},
        "cv_results": {},
        "selected_questions": [],
        "feature_importance": None,
        "best_k": None,
        "ensemble_models": [],
        "hyperparams": {}
    }
    
    # Load test metrics (if exists)
    test_metrics_file = model_dir / "meta_test_metrics.json"
    if test_metrics_file.exists():
        with open(test_metrics_file) as f:
            results["metrics"] = json.load(f)
            results["has_test"] = True
            # Detect task from metrics
            if "macro_f1" in results["metrics"]:
                results["task"] = "classification"
            elif "rmse" in results["metrics"]:
                results["task"] = "regression"
    
    # Load CV results (if exists)
    cv_results_file = model_dir / "cv_results.json"
    if cv_results_file.exists():
        with open(cv_results_file) as f:
            results["cv_results"] = json.load(f)
            results["has_cv"] = True
            if not results["has_test"]:
                if "aggregate_metrics" in results["cv_results"]:
                    if "macro_f1" in results["cv_results"]["aggregate_metrics"]:
                        results["task"] = "classification"
                    elif "rmse" in results["cv_results"]["aggregate_metrics"]:
                        results["task"] = "regression"
    
    # Load ensemble config
    ensemble_file = model_dir / "ensemble_config.json"
    if ensemble_file.exists():
        with open(ensemble_file) as f:
            ensemble_config = json.load(f)
            results["is_ensemble"] = True
            results["ensemble_models"] = ensemble_config.get("ensemble_models", [])
    
    # Load selected questions
    selected_qs_file = model_dir / "selected_questions.csv"
    if selected_qs_file.exists():
        selected_qs = pd.read_csv(selected_qs_file)
        results["selected_questions"] = selected_qs["question_id"].tolist()
    
    # Load question importance
    importance_file = model_dir / "question_embedding_importance.csv"
    if importance_file.exists():
        results["feature_importance"] = pd.read_csv(importance_file)
    
    # Load K selection results
    k_selection_file = model_dir / "cv_k_selection_results.csv"
    if k_selection_file.exists():
        k_df = pd.read_csv(k_selection_file)
        best_k_row = k_df[k_df['is_best'] == True]
        if not best_k_row.empty:
            results["best_k"] = int(best_k_row['k'].iloc[0])
            results["best_k_score"] = best_k_row['mean_cv_score'].iloc[0]
    
    # Load hyperparameters
    hparams_file = model_dir / "best_hyperparams_all_questions.json"
    if hparams_file.exists():
        with open(hparams_file) as f:
            results["hyperparams"] = json.load(f)
    
    return results


# ============================================================================
# Scoring and Comparison Functions
# ============================================================================

def get_primary_score(results: Dict, metric: str = None) -> Tuple[float, str]:
    """Get primary score from results"""
    if results["task"] == "classification":
        if results["has_test"]:
            score = results["metrics"].get("macro_f1", 0.0)
            metric_name = "macro_f1"
        elif results["has_cv"]:
            score = results["cv_results"].get("mean_primary_score", 0.0)
            metric_name = "cv_macro_f1"
        else:
            score = 0.0
            metric_name = "unknown"
    else:  # regression
        if metric == "rmse":
            if results["has_test"]:
                score = -results["metrics"].get("rmse", float('inf'))
                metric_name = "-rmse"
            elif results["has_cv"]:
                score = -results["cv_results"]["aggregate_metrics"].get("rmse", float('inf'))
                metric_name = "-cv_rmse"
            else:
                score = -float('inf')
                metric_name = "unknown"
        elif metric == "r2":
            if results["has_test"]:
                score = results["metrics"].get("r2", -float('inf'))
                metric_name = "r2"
            elif results["has_cv"]:
                score = results["cv_results"]["aggregate_metrics"].get("r2", -float('inf'))
                metric_name = "cv_r2"
        else:  # default for regression
            if results["has_test"]:
                score = -results["metrics"].get("rmse", float('inf'))
                metric_name = "-rmse"
            elif results["has_cv"]:
                score = -results["cv_results"]["aggregate_metrics"].get("rmse", float('inf'))
                metric_name = "-cv_rmse"
            else:
                score = -float('inf')
                metric_name = "unknown"
    
    return score, metric_name


def calculate_confidence_intervals(model1_dir: Path, model2_dir: Path, 
                                   model1_name: str, model2_name: str,
                                   metric: str = "macro_f1", 
                                   n_bootstrap: int = 1000) -> Dict:
    """Calculate bootstrap confidence intervals for model comparison"""
    
    # Load predictions
    preds1_file = model1_dir / "meta_test_predictions.csv"
    preds2_file = model2_dir / "meta_test_predictions.csv"
    
    if not preds1_file.exists() or not preds2_file.exists():
        return {"error": "Predictions files not found"}
    
    preds1 = pd.read_csv(preds1_file)
    preds2 = pd.read_csv(preds2_file)
    
    # Ensure same order
    preds1 = preds1.sort_values('speaker_id').reset_index(drop=True)
    preds2 = preds2.sort_values('speaker_id').reset_index(drop=True)
    
    y_true = preds1['y_true'].values
    y_pred1 = preds1['y_pred'].values
    y_pred2 = preds2['y_pred'].values
    
    # Bootstrap
    np.random.seed(42)
    n_samples = len(y_true)
    scores1 = []
    scores2 = []
    differences = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        if metric == "macro_f1":
            score1 = f1_score(y_true[indices], y_pred1[indices], average='macro', zero_division=0)
            score2 = f1_score(y_true[indices], y_pred2[indices], average='macro', zero_division=0)
        elif metric == "rmse":
            score1 = -np.sqrt(mean_squared_error(y_true[indices], y_pred1[indices]))
            score2 = -np.sqrt(mean_squared_error(y_true[indices], y_pred2[indices]))
        elif metric == "r2":
            score1 = r2_score(y_true[indices], y_pred1[indices])
            score2 = r2_score(y_true[indices], y_pred2[indices])
        else:
            score1 = f1_score(y_true[indices], y_pred1[indices], average='macro', zero_division=0)
            score2 = f1_score(y_true[indices], y_pred2[indices], average='macro', zero_division=0)
        
        scores1.append(score1)
        scores2.append(score2)
        differences.append(score1 - score2)
    
    # Calculate statistics
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    p_value = 2 * min(np.mean(np.array(differences) <= 0), np.mean(np.array(differences) >= 0))
    significantly_different = not (ci_lower <= 0 <= ci_upper)
    
    return {
        "model1_mean": np.mean(scores1),
        "model2_mean": np.mean(scores2),
        "model1_std": np.std(scores1),
        "model2_std": np.std(scores2),
        "diff_mean": np.mean(differences),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": significantly_different
    }


# ============================================================================
# Ranking and Selection Functions
# ============================================================================

def rank_models(all_results: List[Dict], primary_metric: str = "auto", 
                secondary_metrics: List[str] = None) -> pd.DataFrame:
    """Rank models based on multiple criteria"""
    
    if secondary_metrics is None:
        secondary_metrics = []
    
    ranking_data = []
    
    for results in all_results:
        score, metric_name = get_primary_score(results, primary_metric if primary_metric != "auto" else None)
        
        row = {
            "rank": 0,
            "model": results["model_name"],
            "task": results["task"],
            "primary_score": score,
            "primary_metric": metric_name,
            "has_test": results["has_test"],
            "has_cv": results["has_cv"],
            "is_ensemble": results["is_ensemble"],
            "n_ensemble_models": len(results.get("ensemble_models", [])),
            "n_selected_questions": len(results.get("selected_questions", [])),
            "best_k": results.get("best_k", None),
            "best_k_score": results.get("best_k_score", None)
        }
        
        # Add secondary metrics
        if results["task"] == "classification":
            if results["has_test"]:
                row["test_macro_f1"] = results["metrics"].get("macro_f1", None)
                row["test_weighted_f1"] = results["metrics"].get("weighted_f1", None)
                row["test_balanced_acc"] = results["metrics"].get("balanced_accuracy", None)
            if results["has_cv"]:
                row["cv_mean_score"] = results["cv_results"].get("mean_primary_score", None)
                row["cv_std_score"] = results["cv_results"].get("std_primary_score", None)
        else:  # regression
            if results["has_test"]:
                row["test_rmse"] = results["metrics"].get("rmse", None)
                row["test_mae"] = results["metrics"].get("mae", None)
                row["test_r2"] = results["metrics"].get("r2", None)
            if results["has_cv"]:
                cv_metrics = results["cv_results"].get("aggregate_metrics", {})
                row["cv_rmse"] = cv_metrics.get("rmse", None)
                row["cv_r2"] = cv_metrics.get("r2", None)
        
        # Add top questions if available
        if results["feature_importance"] is not None and len(results["feature_importance"]) > 0:
            row["top_question"] = results["feature_importance"].iloc[0]["question_id"]
            row["top_importance"] = results["feature_importance"].iloc[0]["importance"]
        
        ranking_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(ranking_data)
    
    # Sort by primary score
    df = df.sort_values("primary_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    return df


def select_best_models(df: pd.DataFrame, selection_criteria: Dict) -> Dict:
    """Select best models based on custom criteria"""
    
    selections = {}
    
    # 1. Best overall
    selections["best_overall"] = df.iloc[0]["model"]
    
    # 2. Best non-ensemble (simpler model)
    non_ensemble_df = df[df["is_ensemble"] == False]
    if len(non_ensemble_df) > 0:
        selections["best_single_model"] = non_ensemble_df.iloc[0]["model"]
    
    # 3. Most stable (lowest CV std)
    if "cv_std_score" in df.columns:
        stable_idx = df["cv_std_score"].fillna(float('inf')).idxmin()
        selections["most_stable"] = df.loc[stable_idx, "model"]
    
    # 4. Most efficient (fewest questions)
    efficient_idx = df["n_selected_questions"].fillna(float('inf')).idxmin()
    selections["most_efficient"] = df.loc[efficient_idx, "model"]
    
    # 5. Best per task if mixed
    if len(df["task"].unique()) > 1:
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            selections[f"best_{task}"] = task_df.iloc[0]["model"]
    
    # 6. Best with statistical significance (if comparisons provided)
    if "statistical_comparisons" in selection_criteria:
        comparisons = selection_criteria["statistical_comparisons"]
        winners = {}
        for comp in comparisons:
            if comp["significant"] and comp["diff_mean"] > 0:
                winners[comp["model1"]] = winners.get(comp["model1"], 0) + 1
            elif comp["significant"] and comp["diff_mean"] < 0:
                winners[comp["model2"]] = winners.get(comp["model2"], 0) + 1
        
        if winners:
            selections["most_statistically_superior"] = max(winners, key=winners.get)
    
    return selections


# ============================================================================
# Visualization Functions
# ============================================================================

def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for all models"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Primary scores comparison
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(df))]
    bars = ax.barh(df['model'], df['primary_score'], color=colors)
    ax.set_xlabel(df.iloc[0]['primary_metric'])
    ax.set_title('Model Performance Comparison\n(higher is better)')
    ax.axvline(x=df.iloc[0]['primary_score'], color='red', linestyle='--', alpha=0.5, label='Best Model')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, df['primary_score'])):
        ax.text(score, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', ha='left' if score < max(df['primary_score']) else 'right')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Complexity vs Performance
    if 'n_selected_questions' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for _, row in df.iterrows():
            marker = 'o' if not row['is_ensemble'] else 's'
            color = 'red' if row['model'] == df.iloc[0]['model'] else 'blue'
            ax.scatter(row['n_selected_questions'], row['primary_score'], 
                      s=100, marker=marker, color=color, alpha=0.7)
            ax.annotate(row['model'], (row['n_selected_questions'], row['primary_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Number of Selected Questions')
        ax.set_ylabel('Primary Score')
        ax.set_title('Model Complexity vs Performance\n(squares=ensemble, circles=single)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'complexity_vs_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Radar chart for multi-dimensional comparison
    if len(df) <= 10:  # Radar chart gets messy with many models
        metrics_to_plot = []
        if 'test_macro_f1' in df.columns:
            metrics_to_plot.append(('test_macro_f1', 'Macro F1', True))
        if 'test_rmse' in df.columns:
            metrics_to_plot.append(('test_rmse', 'RMSE (neg)', False))
        if 'test_r2' in df.columns:
            metrics_to_plot.append(('test_r2', 'R²', True))
        if 'n_selected_questions' in df.columns:
            metrics_to_plot.append(('n_selected_questions', '# Questions (neg)', False))
        if 'n_ensemble_models' in df.columns:
            metrics_to_plot.append(('n_ensemble_models', '# Models (neg)', False))
        
        if metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Normalize metrics
            n_metrics = len(metrics_to_plot)
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            for _, row in df.iterrows():
                values = []
                for col, _, higher_better in metrics_to_plot:
                    val = row[col]
                    if val is None:
                        val = 0
                    # Normalize to [0,1] across models
                    col_vals = df[col].dropna()
                    if len(col_vals) > 0:
                        if higher_better:
                            val = (val - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-6)
                        else:
                            val = (col_vals.max() - val) / (col_vals.max() - col_vals.min() + 1e-6)
                    values.append(val)
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([label for _, label, _ in metrics_to_plot], size=10)
            ax.set_title('Model Comparison Radar Chart', size=14, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.tight_layout()
            plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Plots saved to {output_dir}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(df: pd.DataFrame, selections: Dict, comparisons: List[Dict], 
                    output_file: Path):
    """Generate comprehensive report"""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL SELECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total models evaluated: {len(df)}\n")
        f.write(f"Classification models: {len(df[df['task'] == 'classification'])}\n")
        f.write(f"Regression models: {len(df[df['task'] == 'regression'])}\n")
        f.write(f"Ensemble models: {len(df[df['is_ensemble'] == True])}\n\n")
        
        # Rankings
        f.write("\nMODEL RANKINGS\n")
        f.write("-"*40 + "\n")
        f.write(df[['rank', 'model', 'primary_score', 'primary_metric', 
                    'n_selected_questions', 'is_ensemble']].to_string(index=False))
        f.write("\n\n")
        
        # Selections
        f.write("\nBEST MODEL SELECTIONS\n")
        f.write("-"*40 + "\n")
        for criterion, model in selections.items():
            if not criterion.startswith("_"):
                f.write(f"{criterion.replace('_', ' ').title()}: {model}\n")
        f.write("\n")
        
        # Detailed metrics for top 3
        f.write("\nTOP 3 MODELS - DETAILED METRICS\n")
        f.write("-"*40 + "\n")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            f.write(f"\n{i+1}. {row['model']}\n")
            f.write(f"   Primary Score: {row['primary_score']:.4f}\n")
            if 'test_macro_f1' in row and pd.notna(row['test_macro_f1']):
                f.write(f"   Test Macro F1: {row['test_macro_f1']:.4f}\n")
            if 'test_rmse' in row and pd.notna(row['test_rmse']):
                f.write(f"   Test RMSE: {row['test_rmse']:.4f}\n")
            if 'test_r2' in row and pd.notna(row['test_r2']):
                f.write(f"   Test R²: {row['test_r2']:.4f}\n")
            if 'n_selected_questions' in row:
                f.write(f"   Selected Questions: {row['n_selected_questions']}\n")
            if row['is_ensemble']:
                f.write(f"   Ensemble Models: {row.get('n_ensemble_models', 'N/A')}\n")
        
        # Statistical comparisons
        if comparisons:
            f.write("\nSTATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-"*40 + "\n")
            for comp in comparisons:
                if "error" not in comp:
                    f.write(f"\n{comp['model1']} vs {comp['model2']}:\n")
                    f.write(f"  Mean difference: {comp['diff_mean']:.4f}\n")
                    f.write(f"  95% CI: [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]\n")
                    f.write(f"  P-value: {comp['p_value']:.4f}\n")
                    if comp['significant']:
                        winner = comp['model1'] if comp['diff_mean'] > 0 else comp['model2']
                        f.write(f"  ✓ {winner} is significantly better\n")
                    else:
                        f.write(f"  ✗ No significant difference\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        f.write(f"🏆 BEST OVERALL: {selections.get('best_overall', 'N/A')}\n")
        if 'best_single_model' in selections:
            f.write(f"📊 BEST SINGLE MODEL (non-ensemble): {selections['best_single_model']}\n")
        if 'most_efficient' in selections:
            f.write(f"⚡ MOST EFFICIENT (fewest questions): {selections['most_efficient']}\n")
        if 'most_stable' in selections:
            f.write(f"📈 MOST STABLE (lowest CV variance): {selections['most_stable']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Report saved to {output_file}")


# ============================================================================
# Main Function with Argparse
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model selection tool for comparing trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - compare all models in results directory
  python select_best_model.py --results-dir ./results
  
  # Compare specific model directories
  python select_best_model.py --model-dirs ./roberta_base ./distilbert_base ./bert_base
  
  # For regression task, use RMSE as primary metric
  python select_best_model.py --results-dir ./results --task regression --metric rmse
  
  # Compare all models with statistical testing
  python select_best_model.py --results-dir ./results --compare-all --statistical-test
  
  # Export results to JSON
  python select_best_model.py --results-dir ./results --export-json best_models.json
  
  # Custom selection criteria
  python select_best_model.py --results-dir ./results --prefer-single --min-questions 5
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--results-dir", type=str, 
                             help="Directory containing all model result subdirectories")
    input_group.add_argument("--model-dirs", nargs="+", type=str,
                             help="Specific model directories to compare")
    
    # Task and metrics
    parser.add_argument("--task", choices=["classification", "regression", "auto"], 
                        default="auto", help="Task type (auto-detect if not specified)")
    parser.add_argument("--metric", type=str, default="auto",
                        choices=["auto", "macro_f1", "rmse", "r2", "balanced_accuracy"],
                        help="Primary metric for comparison")
    
    # Selection criteria
    parser.add_argument("--prefer-single", action="store_true",
                        help="Prefer single models over ensembles")
    parser.add_argument("--prefer-fewer-questions", action="store_true",
                        help="Prefer models that use fewer questions")
    parser.add_argument("--min-questions", type=int, default=None,
                        help="Minimum number of selected questions (filter out models with fewer)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of selected questions (filter out models with more)")
    
    # Comparison options
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all models against each other")
    parser.add_argument("--statistical-test", action="store_true",
                        help="Perform statistical significance tests")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap samples for statistical tests")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./model_selection",
                        help="Directory to save outputs (default: ./model_selection)")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export best model info to JSON file")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model directories
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory {results_dir} does not exist")
            sys.exit(1)
        model_dirs = find_model_dirs(results_dir)
        print(f"Found {len(model_dirs)} model directories in {results_dir}")
    else:
        model_dirs = [Path(d) for d in args.model_dirs]
        print(f"Comparing {len(model_dirs)} specific model directories")
    
    if len(model_dirs) == 0:
        print("Error: No model directories found")
        sys.exit(1)
    
    # Load results for all models
    print("\nLoading model results...")
    all_results = []
    for model_dir in model_dirs:
        if args.verbose:
            print(f"  Loading {model_dir.name}...")
        results = load_model_results(model_dir)
        all_results.append(results)
    
    # Apply filters
    if args.min_questions:
        all_results = [r for r in all_results if len(r.get("selected_questions", [])) >= args.min_questions]
        print(f"After min_questions filter: {len(all_results)} models remain")
    if args.max_questions:
        all_results = [r for r in all_results if len(r.get("selected_questions", [])) <= args.max_questions]
        print(f"After max_questions filter: {len(all_results)} models remain")
    
    if len(all_results) == 0:
        print("Error: No models remain after filtering")
        sys.exit(1)
    
    # Determine primary metric
    primary_metric = args.metric
    if primary_metric == "auto":
        # Detect based on task
        if args.task != "auto":
            if args.task == "classification":
                primary_metric = "macro_f1"
            else:
                primary_metric = "rmse"
        else:
            # Auto-detect from first model
            if all_results[0]["task"] == "classification":
                primary_metric = "macro_f1"
            else:
                primary_metric = "rmse"
    
    # Rank models
    print(f"\nRanking models using primary metric: {primary_metric}")
    ranking_df = rank_models(all_results, primary_metric)
    
    # Print ranking
    print("\n" + "="*80)
    print("MODEL RANKING")
    print("="*80)
    print(ranking_df[['rank', 'model', 'primary_score', 'primary_metric', 
                      'n_selected_questions', 'is_ensemble']].to_string(index=False))
    
    # Statistical comparisons
    comparisons = []
    if args.statistical_test and len(model_dirs) >= 2:
        print("\nPerforming statistical significance tests...")
        
        # Compare top model against others
        top_model = ranking_df.iloc[0]['model']
        top_dir = None
        for r in all_results:
            if r['model_name'] == top_model:
                top_dir = Path(r['model_path'])
                break
        
        for r in all_results:
            if r['model_name'] != top_model:
                other_dir = Path(r['model_path'])
                metric_for_test = "macro_f1" if r["task"] == "classification" else "rmse"
                
                comp = calculate_confidence_intervals(
                    top_dir, other_dir, top_model, r['model_name'],
                    metric=metric_for_test, n_bootstrap=args.n_bootstrap
                )
                comparisons.append(comp)
                
                if args.verbose:
                    if comp.get('significant', False):
                        print(f"  ✓ {top_model} > {r['model_name']} (p={comp['p_value']:.4f})")
                    else:
                        print(f"  ✗ No significant difference between {top_model} and {r['model_name']}")
    
    # Select best models
    selection_criteria = {
        "statistical_comparisons": comparisons if args.compare_all else []
    }
    selections = select_best_models(ranking_df, selection_criteria)
    
    # Print selections
    print("\n" + "="*80)
    print("BEST MODEL SELECTIONS")
    print("="*80)
    for criterion, model in selections.items():
        if not criterion.startswith("_"):
            print(f"  {criterion.replace('_', ' ').title()}: {model}")
    
    # Print detailed info for best model
    best_model = selections['best_overall']
    best_result = next(r for r in all_results if r['model_name'] == best_model)
    
    print("\n" + "="*80)
    print(f"DETAILED INFO FOR BEST MODEL: {best_model}")
    print("="*80)
    
    if best_result["task"] == "classification":
        if best_result["has_test"]:
            print(f"  Test Macro F1: {best_result['metrics'].get('macro_f1', 'N/A'):.4f}")
            print(f"  Test Weighted F1: {best_result['metrics'].get('weighted_f1', 'N/A'):.4f}")
            print(f"  Test Balanced Accuracy: {best_result['metrics'].get('balanced_accuracy', 'N/A'):.4f}")
    else:
        if best_result["has_test"]:
            print(f"  Test RMSE: {best_result['metrics'].get('rmse', 'N/A'):.4f}")
            print(f"  Test MAE: {best_result['metrics'].get('mae', 'N/A'):.4f}")
            print(f"  Test R²: {best_result['metrics'].get('r2', 'N/A'):.4f}")
    
    print(f"  Selected Questions: {len(best_result.get('selected_questions', []))}")
    if best_result.get('best_k'):
        print(f"  Best K (from CV): {best_result['best_k']}")
    if best_result.get('is_ensemble'):
        print(f"  Ensemble Models: {best_result.get('ensemble_models', [])}")
    if best_result.get('feature_importance') is not None and len(best_result['feature_importance']) > 0:
        print(f"  Top 3 Important Questions:")
        for i, (_, row) in enumerate(best_result['feature_importance'].head(3).iterrows(), 1):
            print(f"    {i}. {row['question_id']}: {row['importance']:.4f}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating comparison plots...")
        create_comparison_plots(ranking_df, output_dir)
    
    # Generate report
    report_file = output_dir / "model_selection_report.txt"
    generate_report(ranking_df, selections, comparisons, report_file)
    
    # Export best model to JSON
    if args.export_json:
        best_info = {
            "best_model": best_model,
            "best_model_path": best_result["model_path"],
            "task": best_result["task"],
            "primary_metric": primary_metric,
            "primary_score": float(ranking_df[ranking_df['model'] == best_model]['primary_score'].iloc[0]),
            "metrics": best_result["metrics"],
            "selected_questions": best_result.get("selected_questions", []),
            "is_ensemble": best_result.get("is_ensemble", False),
            "ensemble_models": best_result.get("ensemble_models", []),
            "hyperparameters": best_result.get("hyperparams", {}),
            "all_selections": selections
        }
        
        with open(args.export_json, 'w') as f:
            json.dump(best_info, f, indent=2)
        print(f"\n✓ Best model info exported to {args.export_json}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Ranking saved to: {output_dir / 'model_ranking.csv'}")
    print(f"✓ Report saved to: {report_file}")
    print(f"✓ Best model: {best_model}")
    print(f"✓ Best model path: {best_result['model_path']}")
    
    # Save ranking to CSV
    ranking_df.to_csv(output_dir / "model_ranking.csv", index=False)
    
    return best_result, selections


if __name__ == "__main__":
    best_model, selections = main()

'''

# Basic usage - compare all models in results directory
python select_best_model.py --results-dir ./results

# Compare specific model directories
python select_best_model.py --model-dirs ./roberta_base ./distilbert_base ./bert_base

# For regression task with RMSE
python select_best_model.py --results-dir ./results --task regression --metric rmse

# Prefer single models (non-ensemble) and fewer questions
python select_best_model.py --results-dir ./results --prefer-single --prefer-fewer-questions

# Full comparison with statistical testing
python select_best_model.py --results-dir ./results --compare-all --statistical-test

# Export best model info to JSON
python select_best_model.py --results-dir ./results --export-json best_model.json

# Filter models that use between 3 and 10 questions
python select_best_model.py --results-dir ./results --min-questions 3 --max-questions 10

# Verbose output
python select_best_model.py --results-dir ./results --verbose
'''