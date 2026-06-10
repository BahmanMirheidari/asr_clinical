#!/usr/bin/env python3
"""
Calculate comprehensive metrics for binary classification from predictions CSV file.
Includes Bootstrap 95% Confidence Intervals with 10,000 iterations.
Metrics: AUC, Accuracy, Balanced Accuracy, Macro-F1, Macro-Precision, Macro-Recall,
and per-class metrics (Sensitivity/Specificity/Precision/Recall/F1) for a specified class.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def bootstrap_ci(data, metric_func, n_bootstrap=10000, ci=95, random_state=42):
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        data: Tuple of (y_true, y_pred, y_prob)
        metric_func: Function that takes (y_true, y_pred, y_prob) and returns metric value
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage (default: 95)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with mean, lower_ci, upper_ci, and standard deviation
    """
    np.random.seed(random_state)
    y_true, y_pred, y_prob = data
    n_samples = len(y_true)
    
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]
        
        try:
            metric_value = metric_func(y_true_boot, y_pred_boot, y_prob_boot)
            if not np.isnan(metric_value) and not np.isinf(metric_value):
                bootstrap_metrics.append(metric_value)
        except:
            continue
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Calculate confidence intervals
    alpha = (100 - ci) / 100
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_metric = np.mean(bootstrap_metrics)
    lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
    upper_ci = np.percentile(bootstrap_metrics, upper_percentile)
    std_metric = np.std(bootstrap_metrics)
    
    return {
        'mean': mean_metric,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'std': std_metric,
        'ci_percentage': ci,
        'n_bootstrap': n_bootstrap
    }

def calculate_metrics_with_ci(df, y_true_col='y_true', y_pred_col='y_pred', 
                              prob_col='prob_1', focus_class=1, 
                              n_bootstrap=10000, ci=95, random_state=42):
    """
    Calculate all metrics with bootstrap confidence intervals.
    """
    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values
    y_prob = df[prob_col].values
    
    data_tuple = (y_true, y_pred, y_prob)
    
    # Define metric functions for bootstrap
    def auc_func(yt, yp, yprob):
        return roc_auc_score(yt, yprob)
    
    def accuracy_func(yt, yp, yprob):
        return accuracy_score(yt, yp)
    
    def balanced_accuracy_func(yt, yp, yprob):
        return balanced_accuracy_score(yt, yp)
    
    def macro_precision_func(yt, yp, yprob):
        return precision_score(yt, yp, average='macro', zero_division=0)
    
    def macro_recall_func(yt, yp, yprob):
        return recall_score(yt, yp, average='macro', zero_division=0)
    
    def macro_f1_func(yt, yp, yprob):
        return f1_score(yt, yp, average='macro', zero_division=0)
    
    def weighted_precision_func(yt, yp, yprob):
        return precision_score(yt, yp, average='weighted', zero_division=0)
    
    def weighted_recall_func(yt, yp, yprob):
        return recall_score(yt, yp, average='weighted', zero_division=0)
    
    def weighted_f1_func(yt, yp, yprob):
        return f1_score(yt, yp, average='weighted', zero_division=0)
    
    def mcc_func(yt, yp, yprob):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0
        return ((tp * tn) - (fp * fn)) / denominator
    
    def gmean_func(yt, yp, yprob):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        return np.sqrt(recall_0 * recall_1)
    
    def class_specificity_func(yt, yp, yprob, pos_class):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        if pos_class == 0:
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def class_precision_func(yt, yp, yprob, pos_class):
        return precision_score(yt, yp, pos_label=pos_class, zero_division=0)
    
    def class_recall_func(yt, yp, yprob, pos_class):
        return recall_score(yt, yp, pos_label=pos_class, zero_division=0)
    
    def class_f1_func(yt, yp, yprob, pos_class):
        return f1_score(yt, yp, pos_label=pos_class, zero_division=0)
    
    # Calculate point estimates
    auc_point = auc_func(y_true, y_pred, y_prob)
    accuracy_point = accuracy_func(y_true, y_pred, y_prob)
    balanced_acc_point = balanced_accuracy_func(y_true, y_pred, y_prob)
    macro_prec_point = macro_precision_func(y_true, y_pred, y_prob)
    macro_rec_point = macro_recall_func(y_true, y_pred, y_prob)
    macro_f1_point = macro_f1_func(y_true, y_pred, y_prob)
    weighted_prec_point = weighted_precision_func(y_true, y_pred, y_prob)
    weighted_rec_point = weighted_recall_func(y_true, y_pred, y_prob)
    weighted_f1_point = weighted_f1_func(y_true, y_pred, y_prob)
    mcc_point = mcc_func(y_true, y_pred, y_prob)
    gmean_point = gmean_func(y_true, y_pred, y_prob)
    
    # Calculate bootstrap CIs
    print(f"Calculating bootstrap CIs with {n_bootstrap} iterations... (this may take a moment)")
    
    metrics_with_ci = {
        'auc': {'point': auc_point, **bootstrap_ci(data_tuple, auc_func, n_bootstrap, ci, random_state)},
        'accuracy': {'point': accuracy_point, **bootstrap_ci(data_tuple, accuracy_func, n_bootstrap, ci, random_state)},
        'balanced_accuracy': {'point': balanced_acc_point, **bootstrap_ci(data_tuple, balanced_accuracy_func, n_bootstrap, ci, random_state)},
        'macro_precision': {'point': macro_prec_point, **bootstrap_ci(data_tuple, macro_precision_func, n_bootstrap, ci, random_state)},
        'macro_recall': {'point': macro_rec_point, **bootstrap_ci(data_tuple, macro_recall_func, n_bootstrap, ci, random_state)},
        'macro_f1': {'point': macro_f1_point, **bootstrap_ci(data_tuple, macro_f1_func, n_bootstrap, ci, random_state)},
        'weighted_precision': {'point': weighted_prec_point, **bootstrap_ci(data_tuple, weighted_precision_func, n_bootstrap, ci, random_state)},
        'weighted_recall': {'point': weighted_rec_point, **bootstrap_ci(data_tuple, weighted_recall_func, n_bootstrap, ci, random_state)},
        'weighted_f1': {'point': weighted_f1_point, **bootstrap_ci(data_tuple, weighted_f1_func, n_bootstrap, ci, random_state)},
        'matthews_corrcoef': {'point': mcc_point, **bootstrap_ci(data_tuple, mcc_func, n_bootstrap, ci, random_state)},
        'geometric_mean': {'point': gmean_point, **bootstrap_ci(data_tuple, gmean_func, n_bootstrap, ci, random_state)},
    }
    
    # Per-class metrics
    for class_label in [0, 1]:
        # Specificity requires special handling
        def specificity_func_wrapper(pos_class):
            return lambda yt, yp, yprob: class_specificity_func(yt, yp, yprob, pos_class)
        
        def precision_func_wrapper(pos_class):
            return lambda yt, yp, yprob: class_precision_func(yt, yp, yprob, pos_class)
        
        def recall_func_wrapper(pos_class):
            return lambda yt, yp, yprob: class_recall_func(yt, yp, yprob, pos_class)
        
        def f1_func_wrapper(pos_class):
            return lambda yt, yp, yprob: class_f1_func(yt, yp, yprob, pos_class)
        
        spec_point = class_specificity_func(y_true, y_pred, y_prob, class_label)
        prec_point = class_precision_func(y_true, y_pred, y_prob, class_label)
        rec_point = class_recall_func(y_true, y_pred, y_prob, class_label)
        f1_point = class_f1_func(y_true, y_pred, y_prob, class_label)
        
        metrics_with_ci[f'class_{class_label}'] = {
            'specificity': {'point': spec_point, **bootstrap_ci(data_tuple, specificity_func_wrapper(class_label), n_bootstrap, ci, random_state)},
            'precision': {'point': prec_point, **bootstrap_ci(data_tuple, precision_func_wrapper(class_label), n_bootstrap, ci, random_state)},
            'recall': {'point': rec_point, **bootstrap_ci(data_tuple, recall_func_wrapper(class_label), n_bootstrap, ci, random_state)},
            'f1_score': {'point': f1_point, **bootstrap_ci(data_tuple, f1_func_wrapper(class_label), n_bootstrap, ci, random_state)},
            'support': int(np.sum(y_true == class_label))
        }
    
    # Confusion matrix (point estimate only)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics_with_ci['confusion_matrix'] = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    
    # Prevalence
    metrics_with_ci['prevalence'] = np.mean(y_true)
    
    return metrics_with_ci

def format_ci_text(metric_dict, decimal_places=4):
    """Format metric with CI for display."""
    if metric_dict is None:
        return "N/A"
    point = metric_dict['point']
    lower = metric_dict['lower_ci']
    upper = metric_dict['upper_ci']
    ci_perc = metric_dict['ci_percentage']
    return f"{point:.{decimal_places}f} ({ci_perc}% CI: {lower:.{decimal_places}f}-{upper:.{decimal_places}f})"

def display_metrics_table(metrics_with_ci, focus_class=1, per_fold_df=None):
    """
    Display metrics with confidence intervals.
    """
    print("\n" + "=" * 100)
    print(f"BINARY CLASSIFICATION METRICS WITH {metrics_with_ci['auc']['ci_percentage']}% CONFIDENCE INTERVALS")
    print(f"Bootstrap iterations: {metrics_with_ci['auc']['n_bootstrap']}")
    print("=" * 100)
    
    # OVERALL METRICS
    print("\n📊 OVERALL METRICS:")
    print("-" * 100)
    print(f"  • AUC:                      {format_ci_text(metrics_with_ci['auc'])}")
    print(f"  • Accuracy:                 {format_ci_text(metrics_with_ci['accuracy'])}")
    print(f"  • Balanced Accuracy:        {format_ci_text(metrics_with_ci['balanced_accuracy'])}")
    print(f"  • Macro-F1:                 {format_ci_text(metrics_with_ci['macro_f1'])}")
    print(f"  • Macro-Precision (PR):     {format_ci_text(metrics_with_ci['macro_precision'])}")
    print(f"  • Macro-Recall (RC):        {format_ci_text(metrics_with_ci['macro_recall'])}")
    print(f"  • Weighted-F1:              {format_ci_text(metrics_with_ci['weighted_f1'])}")
    print(f"  • Weighted-Precision:       {format_ci_text(metrics_with_ci['weighted_precision'])}")
    print(f"  • Weighted-Recall:          {format_ci_text(metrics_with_ci['weighted_recall'])}")
    print(f"  • Geometric Mean (G-Mean):  {format_ci_text(metrics_with_ci['geometric_mean'])}")
    print(f"  • MCC:                      {format_ci_text(metrics_with_ci['matthews_corrcoef'])}")
    print(f"  • Prevalence:               {metrics_with_ci['prevalence']:.4f}")
    
    # CONFUSION MATRIX
    print("\n🔢 CONFUSION MATRIX (Point Estimate):")
    print("-" * 100)
    cm = metrics_with_ci['confusion_matrix']
    print(f"  • True Negatives (TN):  {cm['tn']:>5}   |  False Positives (FP):  {cm['fp']:>5}")
    print(f"  • False Negatives (FN): {cm['fn']:>5}   |  True Positives (TP):   {cm['tp']:>5}")
    
    # FOCUS CLASS METRICS
    focus = metrics_with_ci[f'class_{focus_class}']
    print(f"\n🎯 FOCUS CLASS {focus_class} METRICS (Sp/Sn/Pr/Rc/F1):")
    print("-" * 100)
    print(f"  • Sensitivity (Sn / Recall):     {format_ci_text(focus['recall'])}")
    print(f"  • Specificity (Sp):              {format_ci_text(focus['specificity'])}")
    print(f"  • Precision (Pr):                {format_ci_text(focus['precision'])}")
    print(f"  • Recall (Rc):                   {format_ci_text(focus['recall'])}")
    print(f"  • F1-Score:                      {format_ci_text(focus['f1_score'])}")
    print(f"  • Support:                       {focus['support']}")
    
    # OTHER CLASS METRICS
    other_class = 1 - focus_class
    other = metrics_with_ci[f'class_{other_class}']
    print(f"\n📈 OTHER CLASS {other_class} METRICS:")
    print("-" * 100)
    print(f"  • Sensitivity (Sn / Recall):     {format_ci_text(other['recall'])}")
    print(f"  • Specificity (Sp):              {format_ci_text(other['specificity'])}")
    print(f"  • Precision (Pr):                {format_ci_text(other['precision'])}")
    print(f"  • Recall (Rc):                   {format_ci_text(other['recall'])}")
    print(f"  • F1-Score:                      {format_ci_text(other['f1_score'])}")
    print(f"  • Support:                       {other['support']}")
    
    # COMPLETE PER-CLASS TABLE WITH CIs
    print("\n📋 COMPLETE PER-CLASS METRICS (with 95% CI):")
    print("-" * 100)
    
    for class_label in [0, 1]:
        class_metrics = metrics_with_ci[f'class_{class_label}']
        print(f"\n  Class {class_label} (Support: {class_metrics['support']}):")
        print(f"    • Sensitivity (Sn):  {format_ci_text(class_metrics['recall'])}")
        print(f"    • Specificity (Sp):  {format_ci_text(class_metrics['specificity'])}")
        print(f"    • Precision (Pr):    {format_ci_text(class_metrics['precision'])}")
        print(f"    • Recall (Rc):       {format_ci_text(class_metrics['recall'])}")
        print(f"    • F1-Score:          {format_ci_text(class_metrics['f1_score'])}")
    
    print("\n" + "=" * 100)

def save_to_csv_with_ci(metrics_with_ci, output_prefix, focus_class):
    """
    Save metrics with confidence intervals to CSV files.
    """
    # 1. Overall metrics with CI
    overall_metrics = ['auc', 'accuracy', 'balanced_accuracy', 'macro_precision', 
                      'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall',
                      'weighted_f1', 'geometric_mean', 'matthews_corrcoef']
    
    overall_data = []
    for metric in overall_metrics:
        m = metrics_with_ci[metric]
        overall_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Point_Estimate': m['point'],
            f'{m["ci_percentage"]}_CI_Lower': m['lower_ci'],
            f'{m["ci_percentage"]}_CI_Upper': m['upper_ci'],
            'Std_Dev': m['std'],
            'CI_Range': f"[{m['lower_ci']:.4f}, {m['upper_ci']:.4f}]"
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_file = f"{output_prefix}_overall_metrics_with_ci.csv"
    overall_df.to_csv(overall_file, index=False)
    print(f"✅ Saved overall metrics with CI to: {overall_file}")
    
    # 2. Per-class metrics with CI
    class_data = []
    for class_label in [0, 1]:
        class_metrics = metrics_with_ci[f'class_{class_label}']
        for metric_name in ['specificity', 'precision', 'recall', 'f1_score']:
            m = class_metrics[metric_name]
            class_data.append({
                'Class': class_label,
                'Metric': metric_name,
                'Point_Estimate': m['point'],
                f'{m["ci_percentage"]}_CI_Lower': m['lower_ci'],
                f'{m["ci_percentage"]}_CI_Upper': m['upper_ci'],
                'Std_Dev': m['std'],
                'Support': class_metrics['support']
            })
    
    class_df = pd.DataFrame(class_data)
    class_file = f"{output_prefix}_per_class_metrics_with_ci.csv"
    class_df.to_csv(class_file, index=False)
    print(f"✅ Saved per-class metrics with CI to: {class_file}")
    
    # 3. Focus class metrics (detailed)
    focus = metrics_with_ci[f'class_{focus_class}']
    focus_data = []
    for metric_name in ['specificity', 'precision', 'recall', 'f1_score']:
        m = focus[metric_name]
        focus_data.append({
            'Metric': metric_name,
            'Point_Estimate': m['point'],
            f'{m["ci_percentage"]}_CI_Lower': m['lower_ci'],
            f'{m["ci_percentage"]}_CI_Upper': m['upper_ci'],
            'Std_Dev': m['std']
        })
    
    focus_df = pd.DataFrame(focus_data)
    focus_file = f"{output_prefix}_focus_class_{focus_class}_metrics_with_ci.csv"
    focus_df.to_csv(focus_file, index=False)
    print(f"✅ Saved focus class metrics with CI to: {focus_file}")
    
    # 4. Confusion matrix
    cm = metrics_with_ci['confusion_matrix']
    cm_df = pd.DataFrame({
        '': ['Actual_Negative', 'Actual_Positive'],
        'Predicted_Negative': [cm['tn'], cm['fn']],
        'Predicted_Positive': [cm['fp'], cm['tp']]
    })
    cm_file = f"{output_prefix}_confusion_matrix.csv"
    cm_df.to_csv(cm_file, index=False)
    print(f"✅ Saved confusion matrix to: {cm_file}")
    
    # 5. Summary table with formatted CIs
    summary_data = []
    for metric in overall_metrics:
        m = metrics_with_ci[metric]
        summary_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Value (95% CI)': f"{m['point']:.4f} [{m['lower_ci']:.4f}-{m['upper_ci']:.4f}]"
        })
    
    for class_label in [0, 1]:
        class_metrics = metrics_with_ci[f'class_{class_label}']
        for metric_name in ['recall', 'specificity', 'precision', 'f1_score']:
            m = class_metrics[metric_name]
            summary_data.append({
                'Metric': f"Class {class_label} - {metric_name}",
                'Value (95% CI)': f"{m['point']:.4f} [{m['lower_ci']:.4f}-{m['upper_ci']:.4f}]"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_prefix}_summary_with_ci.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved summary with CI to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate binary classification metrics with Bootstrap 95% CI (10,000 iterations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Focus on class 1 with default 10,000 bootstrap iterations:
  python calculate_metrics_with_ci.py -p predictions.csv
  
  # Focus on class 0 with custom iterations:
  python calculate_metrics_with_ci.py -p predictions.csv --focus_class 0 --n_bootstrap 5000
  
  # Save to CSV files:
  python calculate_metrics_with_ci.py -p predictions.csv --output_csv results --focus_class 1
  
  # Change confidence level to 99%:
  python calculate_metrics_with_ci.py -p predictions.csv --ci 99 --output_csv results
        """
    )
    
    parser.add_argument('--predictions_file', '-p', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--focus_class', '-f', type=int, choices=[0, 1], default=1,
                        help='Class to focus on for detailed Sp/Sn/Pr/Rc/F1 metrics (default: 1)')
    parser.add_argument('--n_bootstrap', '-n', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000)')
    parser.add_argument('--ci', type=int, default=95,
                        help='Confidence interval percentage (default: 95)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_csv', '-c', type=str, default=None,
                        help='Output CSV file prefix (e.g., "results" creates multiple CSV files)')
    parser.add_argument('--output_json', '-j', type=str, default=None,
                        help='Output JSON file (optional)')
    parser.add_argument('--prob_col', type=str, default='prob_1',
                        help='Column name for positive class probabilities (default: prob_1)')
    parser.add_argument('--y_true_col', type=str, default='y_true',
                        help='Column name for true labels (default: y_true)')
    parser.add_argument('--y_pred_col', type=str, default='y_pred',
                        help='Column name for predicted labels (default: y_pred)')
    parser.add_argument('--no_display', action='store_true',
                        help='Suppress console display (only save to files)')
    
    args = parser.parse_args()
    
    # Read predictions file
    try:
        df = pd.read_csv(args.predictions_file)
        print(f"Loaded {len(df)} predictions from {args.predictions_file}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check required columns
    required_cols = [args.y_true_col, args.y_pred_col, args.prob_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Calculate metrics with confidence intervals
    metrics_with_ci = calculate_metrics_with_ci(
        df, 
        y_true_col=args.y_true_col, 
        y_pred_col=args.y_pred_col,
        prob_col=args.prob_col,
        focus_class=args.focus_class,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        random_state=args.random_seed
    )
    
    # Display results
    if not args.no_display:
        display_metrics_table(metrics_with_ci, focus_class=args.focus_class)
    
    # Save to CSV if requested
    if args.output_csv:
        save_to_csv_with_ci(metrics_with_ci, args.output_csv, args.focus_class)
    
    # Save to JSON if requested
    if args.output_json:
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.int32, np.int64):
                return int(obj)
            return obj
        
        # Simplify metrics for JSON
        json_output = {}
        for key, value in metrics_with_ci.items():
            if key in ['auc', 'accuracy', 'balanced_accuracy', 'macro_precision', 
                      'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall',
                      'weighted_f1', 'matthews_corrcoef', 'geometric_mean']:
                json_output[key] = {
                    'point': convert_to_serializable(value['point']),
                    'lower_ci': convert_to_serializable(value['lower_ci']),
                    'upper_ci': convert_to_serializable(value['upper_ci']),
                    'std': convert_to_serializable(value['std']),
                    'ci_percentage': value['ci_percentage'],
                    'n_bootstrap': value['n_bootstrap']
                }
            elif key.startswith('class_'):
                json_output[key] = {}
                for metric_name, metric_value in value.items():
                    if metric_name in ['specificity', 'precision', 'recall', 'f1_score']:
                        json_output[key][metric_name] = {
                            'point': convert_to_serializable(metric_value['point']),
                            'lower_ci': convert_to_serializable(metric_value['lower_ci']),
                            'upper_ci': convert_to_serializable(metric_value['upper_ci']),
                            'std': convert_to_serializable(metric_value['std'])
                        }
                    else:
                        json_output[key][metric_name] = convert_to_serializable(metric_value)
            elif key == 'confusion_matrix':
                json_output[key] = value
            elif key == 'prevalence':
                json_output[key] = convert_to_serializable(value)
        
        with open(args.output_json, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"✅ Metrics saved to {args.output_json}")

if __name__ == "__main__":
    main()