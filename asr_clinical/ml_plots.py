import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate plots from model predictions')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file with columns: speaker_id,y_true,y_pred,fold')
    parser.add_argument('--classes', type=str, required=True,
                        help='Path to speaker classes CSV file with columns: speaker_id,class')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder to save plots')
    parser.add_argument('--task', type=str, choices=['regression', 'classification', 'auto'], 
                        default='auto', help='Type of task (default: auto-detect)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help='List of class names for classification (e.g., --class_names class0 class1 class2)')
    return parser.parse_args()

def load_data(predictions_path, classes_path):
    """Load and merge predictions and class information"""
    pred_df = pd.read_csv(predictions_path)
    class_df = pd.read_csv(classes_path)
    
    # Merge on speaker_id
    merged_df = pred_df.merge(class_df, on='speaker_id', how='left')
    
    return merged_df

def detect_task_type(df):
    """Auto-detect if it's regression or classification"""
    # Check if y_true contains only integers and limited unique values
    unique_true = df['y_true'].nunique()
    unique_pred = df['y_pred'].nunique()
    
    # If both true and pred are integers with few unique values, likely classification
    is_integer_true = df['y_true'].astype(float).fillna(0).apply(lambda x: x.is_integer()).all()
    is_integer_pred = df['y_pred'].astype(float).fillna(0).apply(lambda x: x.is_integer()).all()
    
    if is_integer_true and is_integer_pred and unique_true <= 20 and unique_pred <= 20:
        return 'classification'
    return 'regression'

def prepare_classification_data(df):
    """Prepare data for classification metrics"""
    # Convert to integer if they're floats
    df['y_true_int'] = df['y_true'].astype(int)
    df['y_pred_int'] = df['y_pred'].round().astype(int)
    
    # Get unique classes
    classes = sorted(df['y_true_int'].unique())
    n_classes = len(classes)
    
    return df, classes, n_classes

# ============ REGRESSION PLOTS ============

def regression_plot_1_scatter_with_identity(df, output_path):
    """Plot 1: Actual vs Predicted scatter plot with identity line"""
    plt.figure(figsize=(10, 8))
    
    if 'class' in df.columns and df['class'].notna().any():
        classes = df['class'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for cls, color in zip(classes, colors):
            subset = df[df['class'] == cls]
            plt.scatter(subset['y_true'], subset['y_pred'], 
                       label=cls, alpha=0.6, s=50, color=color)
    else:
        plt.scatter(df['y_true'], df['y_pred'], alpha=0.6, s=50)
    
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted Values (Regression)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'regression_1_actual_vs_predicted.png'), dpi=300)
    plt.close()

def regression_plot_2_residuals(df, output_path):
    """Plot 2: Residual plot"""
    df['error'] = df['y_pred'] - df['y_true']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(df['y_pred'], df['error'], alpha=0.6, s=50)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals (Pred - Actual)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df['error'], bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'regression_2_residual_plots.png'), dpi=300)
    plt.close()

def regression_plot_3_fold_comparison(df, output_path):
    """Plot 3: Performance by fold"""
    if 'fold' not in df.columns:
        return
    
    df['abs_error'] = np.abs(df['y_pred'] - df['y_true'])
    df['squared_error'] = (df['y_pred'] - df['y_true']) ** 2
    
    fold_metrics = df.groupby('fold').agg({
        'abs_error': 'mean',
        'squared_error': 'mean'
    }).reset_index()
    fold_metrics['rmse'] = np.sqrt(fold_metrics['squared_error'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].bar(fold_metrics['fold'], fold_metrics['abs_error'], 
                color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('MAE by Fold', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(fold_metrics['fold'], fold_metrics['rmse'], 
                color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Root Mean Square Error', fontsize=12)
    axes[1].set_title('RMSE by Fold', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'regression_3_fold_comparison.png'), dpi=300)
    plt.close()

def regression_plot_4_class_comparison(df, output_path):
    """Plot 4: Performance comparison by speaker class"""
    if 'class' not in df.columns or df['class'].isna().all():
        return
    
    df['abs_error'] = np.abs(df['y_pred'] - df['y_true'])
    df['error'] = df['y_pred'] - df['y_true']
    
    class_metrics = df.groupby('class').agg({
        'abs_error': ['mean', 'std'],
        'error': ['mean', 'std']
    }).round(4)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    classes = class_metrics.index
    mae_means = class_metrics['abs_error']['mean']
    mae_stds = class_metrics['abs_error']['std']
    
    axes[0].bar(classes, mae_means, yerr=mae_stds, capsize=5,
                color=['#FF9999', '#66B2FF'], edgecolor='black')
    axes[0].set_xlabel('Speaker Class', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('MAE by Speaker Class', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    bias_means = class_metrics['error']['mean']
    bias_stds = class_metrics['error']['std']
    
    axes[1].bar(classes, bias_means, yerr=bias_stds, capsize=5,
                color=['#99CC99', '#FFCC99'], edgecolor='black')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Speaker Class', fontsize=12)
    axes[1].set_ylabel('Mean Bias (Pred - Actual)', fontsize=12)
    axes[1].set_title('Prediction Bias by Speaker Class', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'regression_4_class_comparison.png'), dpi=300)
    plt.close()
    
    return class_metrics

# ============ CLASSIFICATION PLOTS ============

def classification_plot_confusion_matrix(df, classes, output_path, class_names=None):
    """Plot 1: Confusion Matrix"""
    from sklearn.metrics import ConfusionMatrixDisplay
    
    cm = confusion_matrix(df['y_true_int'], df['y_pred_int'], labels=classes)
    
    # Use provided class names or default to class numbers
    display_labels = class_names if class_names else [str(c) for c in classes]
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'classification_1_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Also save normalized version
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=display_labels)
    disp_norm.plot(cmap='Blues', values_format='.2f', ax=plt.gca())
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'classification_1_confusion_matrix_normalized.png'), dpi=300)
    plt.close()
    
    return cm

def classification_plot_metrics_per_class(df, classes, output_path, class_names=None):
    """Plot 2: Per-class metrics (Precision, Recall, F1)"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        df['y_true_int'], df['y_pred_int'], labels=classes
    )
    
    display_labels = class_names if class_names else [str(c) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'classification_2_per_class_metrics.png'), dpi=300)
    plt.close()
    
    return precision, recall, f1, support

def classification_plot_roc_curves(df, classes, output_path, class_names=None):
    """Plot 3: ROC Curves (for multi-class)"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(classes)
    
    # Check if we have probabilities (y_pred might be class indices or probabilities)
    # If y_pred contains class indices, we can't do ROC
    if df['y_pred'].dtype in ['int64', 'int32'] or df['y_pred'].nunique() <= n_classes:
        print("  ℹ Skipping ROC curves (predicted classes are discrete, need probabilities)")
        return None
    
    # Try to interpret y_pred as probabilities
    # For multi-class, we need one probability per class
    if n_classes > 2:
        # Assume y_pred contains class probabilities (might be comma-separated?)
        print("  ⚠ Multi-class ROC requires probability outputs. Skipping.")
        return None
    
    # Binary classification ROC
    y_true_bin = label_binarize(df['y_true_int'], classes=classes)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true_bin, df['y_pred'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'classification_3_roc_curve.png'), dpi=300)
        plt.close()
        
        return roc_auc
    return None

def classification_plot_accuracy_by_fold(df, output_path):
    """Plot 4: Accuracy by fold"""
    if 'fold' not in df.columns:
        return
    
    df['correct'] = (df['y_true_int'] == df['y_pred_int']).astype(int)
    fold_accuracy = df.groupby('fold')['correct'].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(fold_accuracy.index, fold_accuracy.values, 
                   color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, fold_accuracy.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Classification Accuracy by Fold', fontsize=14)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'classification_4_accuracy_by_fold.png'), dpi=300)
    plt.close()

def classification_plot_class_distribution(df, classes, output_path, class_names=None):
    """Plot 5: Class distribution in predictions vs ground truth"""
    display_labels = class_names if class_names else [str(c) for c in classes]
    
    true_counts = df['y_true_int'].value_counts().reindex(classes, fill_value=0)
    pred_counts = df['y_pred_int'].value_counts().reindex(classes, fill_value=0)
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, true_counts.values, width, label='Ground Truth', 
           color='skyblue', edgecolor='black')
    ax.bar(x + width/2, pred_counts.values, width, label='Predictions', 
           color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution: Ground Truth vs Predictions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'classification_5_class_distribution.png'), dpi=300)
    plt.close()

def classification_plot_misclassification_analysis(df, classes, output_path, class_names=None):
    """Plot 6: Misclassification analysis heatmap"""
    # Calculate misclassification matrix
    misclass_matrix = np.zeros((len(classes), len(classes)))
    for true_class in classes:
        for pred_class in classes:
            if true_class != pred_class:
                misclass_matrix[classes.index(true_class), classes.index(pred_class)] = \
                    len(df[(df['y_true_int'] == true_class) & (df['y_pred_int'] == pred_class)])
    
    # Only plot if there are misclassifications
    if misclass_matrix.sum() > 0:
        display_labels = class_names if class_names else [str(c) for c in classes]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=display_labels, yticklabels=display_labels)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Misclassification Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'classification_6_misclassification_heatmap.png'), dpi=300)
        plt.close()

# ============ SUMMARY REPORTS ============

def save_regression_report(df, metrics, class_metrics, output_path):
    """Save regression summary report"""
    with open(os.path.join(output_path, 'regression_summary_report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REGRESSION MODEL PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n\nSTATISTICS BY FOLD:\n")
        f.write("-" * 30 + "\n")
        if 'fold' in df.columns:
            df['abs_error'] = np.abs(df['y_pred'] - df['y_true'])
            df['squared_error'] = (df['y_pred'] - df['y_true']) ** 2
            fold_stats = df.groupby('fold').agg({
                'y_true': 'count',
                'abs_error': 'mean',
                'squared_error': 'mean'
            }).round(4)
            fold_stats['rmse'] = np.sqrt(fold_stats['squared_error'])
            f.write(fold_stats.to_string())
        
        if 'class' in df.columns and not df['class'].isna().all():
            f.write("\n\n\nSTATISTICS BY SPEAKER CLASS:\n")
            f.write("-" * 30 + "\n")
            f.write(class_metrics.to_string())
            
            f.write("\n\n\nBIAS ANALYSIS BY CLASS:\n")
            f.write("-" * 30 + "\n")
            df['error'] = df['y_pred'] - df['y_true']
            for cls in df['class'].unique():
                subset = df[df['class'] == cls]
                bias = subset['error'].mean()
                direction = "overpredicting" if bias > 0 else "underpredicting"
                f.write(f"{cls}: {direction} by {abs(bias):.4f} on average\n")
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Unique speakers: {df['speaker_id'].nunique()}\n")

def save_classification_report(df, classes, class_names, precision, recall, f1, support, cm, output_path):
    """Save classification summary report"""
    from sklearn.metrics import classification_report, accuracy_score
    
    display_labels = class_names if class_names else [str(c) for c in classes]
    accuracy = accuracy_score(df['y_true_int'], df['y_pred_int'])
    
    with open(os.path.join(output_path, 'classification_summary_report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION MODEL PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 40 + "\n")
        for i, cls in enumerate(display_labels):
            f.write(f"{str(cls):<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}\n")
        
        f.write("\n\nCLASSIFICATION REPORT (Detailed):\n")
        f.write("-" * 40 + "\n")
        report = classification_report(df['y_true_int'], df['y_pred_int'], 
                                      target_names=display_labels, zero_division=0)
        f.write(report)
        
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        f.write("True \\ Predicted")
        for cls in display_labels:
            f.write(f"{str(cls):>10}")
        f.write("\n")
        for i, true_cls in enumerate(display_labels):
            f.write(f"{str(true_cls):<15}")
            for j in range(len(display_labels)):
                f.write(f"{cm[i, j]:>10}")
            f.write("\n")
        
        f.write("\n\nSTATISTICS BY FOLD:\n")
        f.write("-" * 30 + "\n")
        if 'fold' in df.columns:
            df['correct'] = (df['y_true_int'] == df['y_pred_int']).astype(int)
            fold_acc = df.groupby('fold')['correct'].agg(['mean', 'count']).round(4)
            fold_acc['mean'] = fold_acc['mean'] * 100
            fold_acc.columns = ['Accuracy (%)', 'Samples']
            f.write(fold_acc.to_string())
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Unique speakers: {df['speaker_id'].nunique()}\n")
        f.write(f"Number of classes: {len(classes)}\n")

# ============ MAIN FUNCTION ============

def main():
    args = parse_arguments()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data(args.predictions, args.classes)
    
    # Detect task type
    if args.task == 'auto':
        task_type = detect_task_type(df)
    else:
        task_type = args.task
    
    print(f"\n📊 Detected task type: {task_type.upper()}")
    print(f"Loaded {len(df)} predictions for {df['speaker_id'].nunique()} speakers")
    print(f"Speaker classes found: {df['class'].unique()}")
    
    if task_type == 'regression':
        print("\n" + "="*50)
        print("Generating REGRESSION plots...")
        print("="*50)
        
        # Calculate basic metrics
        df['error'] = df['y_pred'] - df['y_true']
        df['abs_error'] = np.abs(df['error'])
        df['squared_error'] = df['error'] ** 2
        
        metrics = {
            'MAE': df['abs_error'].mean(),
            'MSE': df['squared_error'].mean(),
            'RMSE': np.sqrt(df['squared_error'].mean()),
            'R²': 1 - (df['squared_error'].sum() / np.sum((df['y_true'] - df['y_true'].mean()) ** 2))
        }
        
        print("\nGenerating plots...")
        regression_plot_1_scatter_with_identity(df, args.output)
        print("  ✓ Plot 1: Actual vs Predicted scatter plot")
        
        regression_plot_2_residuals(df, args.output)
        print("  ✓ Plot 2: Residual plots")
        
        regression_plot_3_fold_comparison(df, args.output)
        print("  ✓ Plot 3: Fold comparison")
        
        class_metrics = regression_plot_4_class_comparison(df, args.output)
        print("  ✓ Plot 4: Class comparison")
        
        save_regression_report(df, metrics, class_metrics, args.output)
        print("  ✓ Summary report saved")
        
        # Print metrics
        print("\n📈 Overall Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    else:  # classification
        print("\n" + "="*50)
        print("Generating CLASSIFICATION plots...")
        print("="*50)
        
        # Prepare classification data
        df, classes, n_classes = prepare_classification_data(df)
        
        print(f"  Number of classes: {n_classes}")
        print(f"  Classes: {classes}")
        
        # Handle class names
        class_names = args.class_names
        if class_names and len(class_names) != len(classes):
            print(f"  ⚠ Warning: Provided {len(class_names)} class names but found {len(classes)} classes. Using default names.")
            class_names = None
        
        print("\nGenerating plots...")
        
        # Generate all classification plots
        cm = classification_plot_confusion_matrix(df, classes, args.output, class_names)
        print("  ✓ Plot 1: Confusion Matrix (raw & normalized)")
        
        precision, recall, f1, support = classification_plot_metrics_per_class(
            df, classes, args.output, class_names
        )
        print("  ✓ Plot 2: Per-class metrics (Precision, Recall, F1)")
        
        classification_plot_roc_curves(df, classes, args.output, class_names)
        print("  ✓ Plot 3: ROC Curve (if applicable)")
        
        classification_plot_accuracy_by_fold(df, args.output)
        print("  ✓ Plot 4: Accuracy by fold")
        
        classification_plot_class_distribution(df, classes, args.output, class_names)
        print("  ✓ Plot 5: Class distribution comparison")
        
        classification_plot_misclassification_analysis(df, classes, args.output, class_names)
        print("  ✓ Plot 6: Misclassification heatmap")
        
        save_classification_report(df, classes, class_names, precision, recall, f1, support, cm, args.output)
        print("  ✓ Summary report saved")
        
        # Print quick metrics
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(df['y_true_int'], df['y_pred_int'])
        print(f"\n📈 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n✅ All plots and reports saved to: {args.output}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(args.output)):
        print(f"  - {file}")

if __name__ == "__main__":
    main()