from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification

from .config import TrainConfig
from .data import load_examples
from .model import load_tokenizer
from .train import choose_device, saved_model_exists, train_one_fold
import shutil
from pathlib import Path


import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial

def hyperparameter_search_optuna(
    train_df: pd.DataFrame,
    split_manager: SplitManager,
    args,
    metadata: dict,
    test_df: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """
    Perform hyperparameter search using Optuna (Bayesian optimization).
    Learns from previous trials and prunes unpromising ones.
    """
    print("Starting Optuna hyperparameter search (Bayesian optimization + pruning)")
    
    # Validate that train_df has required columns
    if 'question_id' not in train_df.columns:
        raise KeyError(f"train_df missing 'question_id' column. Available: {train_df.columns.tolist()}")
    
    # Choose representative question
    candidate_questions = [q.upper() for q in args.questions]
    rep_question = None
    for q in candidate_questions:
        q_train = train_df[train_df["question_id"] == q]
        if len(q_train) >= 20:
            rep_question = q
            break
    
    if rep_question is None:
        print("No question with enough training data. Using defaults.")
        return get_default_params(args)
    
    print(f"Using question '{rep_question}' for hyperparameter search")
    
    # Determine metric and direction
    if args.task == "classification":
        metric_name = "macro_f1"
        direction = "maximize"
    else:
        metric_name = getattr(args, 'regression_metric', 'rmse')
        direction = "minimize"
    
    print(f"Optimizing {metric_name} ({direction})")
    
    # Get fold splits (create once, reuse for all trials)
    folds = split_manager.get_fold_splits(train_df, test_df)
    if not folds:
        print("No folds created. Using defaults.")
        return get_default_params(args)
    
    # Use only the number of folds specified for HPO
    folds = folds[:args.hpo_folds]
    
    # Create Optuna study
    sampler = TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"{args.task}_hpo",
        load_if_exists=True
    )
    
    # Define objective function with partial to pass fixed arguments
    objective_partial = partial(
        objective_function,
        train_df=train_df,
        folds=folds,
        rep_question=rep_question,
        args=args,
        metadata=metadata,
        metric_name=metric_name,
        direction=direction
    )
    
    # Run optimization
    print(f"\nRunning Optuna for {args.hpo_n_trials} trials with {len(folds)}-fold CV")
    study.optimize(
        objective_partial,
        n_trials=args.hpo_n_trials,
        timeout=args.hpo_timeout,  # Optional: stop after N seconds
        show_progress_bar=True,
        n_jobs=1  # Optuna doesn't parallelize well with GPU models
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== Optuna Search Complete ===")
    print(f"Best {metric_name}: {best_value:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Show optimization history
    print("\nOptimization history:")
    for trial in study.trials[-10:]:  # Last 10 trials
        if trial.value is not None:
            print(f"  Trial {trial.number}: {trial.value:.4f} - {trial.params}")
    
    # Save study for later analysis
    study_path = Path(args.output_dir) / "optuna_study.pkl"
    joblib.dump(study, study_path)
    print(f"Saved Optuna study to {study_path}")
    
    # Plot optimization history (optional)
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig1.write_image(Path(args.output_dir) / "optuna_history.png")
        
        fig2 = plot_param_importances(study)
        fig2.write_image(Path(args.output_dir) / "optuna_importance.png")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Add fixed parameters that weren't tuned
    best_params.update({
        "max_length": args.max_length,  # Or could be tuned
        "weight_decay": best_params.get("weight_decay", args.weight_decay),
        "warmup_ratio": best_params.get("warmup_ratio", args.warmup_ratio),
    })
    
    return best_params


def objective_function(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    folds: list,
    rep_question: str,
    args,
    metadata: dict,
    metric_name: str,
    direction: str
) -> float:
    """
    Objective function for Optuna to optimize.
    Returns the metric to optimize (higher is better for maximization, lower for minimization).
    """
    # Suggest hyperparameters with appropriate distributions
    params = {
        # Learning rate: log scale works with positive values
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        
        # Batch size: categorical is fine
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        
        # Epochs: integer range
        "epochs": trial.suggest_int("epochs", 2, 8),
        
        # Weight decay: USE REGULAR (not log) when including zero
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        # OR use log scale with a small epsilon to avoid zero
        # "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
        
        # Warmup ratio: regular scale is fine (can be zero)
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        
        # Max length: categorical
        "max_length": trial.suggest_categorical("max_length", [128, 256, 384, 512]),
    }
    
    # Optional: Add conditional parameters for classification
    if args.task == "classification":
        params["class_weights"] = trial.suggest_categorical("class_weights", ["none", "balanced"])
        if args.loss == "focal":
            params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    
    print(f"\nTrial {trial.number}: testing {params}")
    
    # Evaluate on all folds
    fold_scores = []
    fold_metrics = []
    
    for fold_idx, (fold_train, fold_val) in enumerate(folds):
        # Filter for representative question
        q_fold_train = fold_train[fold_train["question_id"] == rep_question].reset_index(drop=True)
        q_fold_val = fold_val[fold_val["question_id"] == rep_question].reset_index(drop=True)
        
        if len(q_fold_train) == 0 or len(q_fold_val) == 0:
            continue
        
        # Create temp config
        temp_out = Path(args.output_dir) / "temp_hpo_optuna" / f"trial{trial.number}_fold{fold_idx}"
        temp_cfg = TrainConfig(
            asr_file=args.asr_file,
            demo_file=args.demo_file,
            target_column=args.target_column,
            task=args.task,
            output_dir=str(temp_out),
            model_name=args.model_name,
            text_mode="question",
            aggregate_level="speaker",
            num_folds=1,
            test_size=0.0,
            final_dev_size=0.0,
            seed=args.seed + trial.number + fold_idx,
            max_length=params["max_length"],
            batch_size=params["batch_size"],
            eval_batch_size=params["batch_size"],
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            patience=args.patience,
            class_weights=params.get("class_weights", args.class_weights),
            loss=args.loss,
            focal_gamma=params.get("focal_gamma", args.focal_gamma),
            filter_questions=[rep_question],
            min_text_chars=args.min_text_chars,
        )
        
        # Train and evaluate
        metrics = _train_and_evaluate(
            q_fold_train, q_fold_val, temp_cfg, metadata,
            regression_metric=metric_name
        )
        
        if metrics is not None:
            score = metrics[metric_name]
            fold_scores.append(score)
            fold_metrics.append(metrics)
            
            # Report intermediate value for pruning
            trial.report(np.mean(fold_scores), fold_idx)
            
            # Handle pruning
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at fold {fold_idx}")
                raise optuna.TrialPruned()
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(temp_out)
        except:
            pass
    
    if not fold_scores:
        return float('inf') if direction == "minimize" else float('-inf')
    
    # Return the average score
    avg_score = np.mean(fold_scores)
    print(f"Trial {trial.number} complete: {metric_name}={avg_score:.4f} (±{np.std(fold_scores):.4f})")
    
    return avg_score


def get_default_params(args):
    """Return default hyperparameters"""
    return {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "regression_metric": getattr(args, 'regression_metric', 'rmse'),
    }


def cleanup_temp_dirs(temp_dir: Path, keep_best: bool = True, best_params: dict = None):
    """
    Clean up temporary directories created during hyperparameter search.
    
    Args:
        temp_dir: Root directory containing temp_hpo folders
        keep_best: Whether to keep the best performing model's directory
        best_params: Parameters of the best model to keep
    """
    if not temp_dir.exists():
        return
    
    temp_hpo_dir = temp_dir / "temp_hpo"
    if not temp_hpo_dir.exists():
        return
    
    print(f"\nCleaning up temporary hyperparameter search directories...")
    
    # Keep track of which directories to keep
    keep_dirs = set()
    if keep_best and best_params:
        # Construct the pattern for best params
        best_pattern = f"lr{best_params['learning_rate']}_bs{best_params['batch_size']}_ep{best_params['epochs']}_wd{best_params['weight_decay']}_wr{best_params['warmup_ratio']}_ml{best_params['max_length']}"
        keep_dirs.add(best_pattern)
    
    # Count directories for reporting
    total_size = 0
    deleted_count = 0
    kept_count = 0
    
    # Iterate through all question directories
    for question_dir in temp_hpo_dir.iterdir():
        if not question_dir.is_dir():
            continue
            
        # Iterate through parameter combinations
        for param_dir in question_dir.iterdir():
            if not param_dir.is_dir():
                continue
                
            # Check if this directory should be kept
            should_keep = False
            for keep_pattern in keep_dirs:
                if keep_pattern in param_dir.name:
                    should_keep = True
                    break
            
            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in param_dir.rglob('*') if f.is_file())
            total_size += dir_size
            
            if should_keep:
                kept_count += 1
                print(f"  Keeping: {param_dir.name} ({dir_size / 1024 / 1024:.2f} MB)")
            else:
                try:
                    shutil.rmtree(param_dir)
                    deleted_count += 1
                    print(f"  Deleted: {param_dir.name} ({dir_size / 1024 / 1024:.2f} MB)")
                except Exception as e:
                    print(f"  Error deleting {param_dir.name}: {e}")
    
    print(f"\nCleanup complete:")
    print(f"  - Deleted: {deleted_count} directories ({total_size / 1024 / 1024:.2f} MB)")
    print(f"  - Kept: {kept_count} directories")
    
    # Optionally, also clean up empty question directories
    for question_dir in temp_hpo_dir.iterdir():
        if question_dir.is_dir() and not any(question_dir.iterdir()):
            try:
                question_dir.rmdir()
                print(f"  Removed empty directory: {question_dir.name}")
            except:
                pass

# ----------------------------------------------------------------------
#  Split management (creates / loads CSV files)
# ----------------------------------------------------------------------
class SplitManager:
    def __init__(self, splits_dir: Path, task: str, train_frac: float, val_frac: float,
                 test_frac: float, seed: int, n_folds: int = 5):
        self.splits_dir = Path(splits_dir)
        self.task = task
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.n_folds = n_folds
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def get_final_splits(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load or create final_train/val/test CSV files."""
        train_path = self.splits_dir / "final_train.csv"
        val_path = self.splits_dir / "final_val.csv"
        test_path = self.splits_dir / "final_test.csv"

        if train_path.exists() and val_path.exists() and test_path.exists():
            print("Loading existing final splits.")
            return (
                pd.read_csv(train_path),
                pd.read_csv(val_path),
                pd.read_csv(test_path),
            )

        print("Creating final train/val/test splits (by speaker).")
        trainval_idx, test_idx = self._speaker_split(df, self.test_frac, self.seed)
        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        rel_val_frac = self.val_frac / (self.train_frac + self.val_frac)
        train_idx, val_idx = self._speaker_split(trainval_df, rel_val_frac, self.seed + 1)
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        for df_out, path in zip([train_df, val_df, test_df],
                                [train_path, val_path, test_path]):
            df_out.to_csv(path, index=False)
        print("Final splits saved.")
        return train_df, val_df, test_df

    def get_fold_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Load or create fold splits for inner CV."""
        # Check if we need to create folds
        need_create = False
        for fold_idx in range(self.n_folds):
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            if not (train_path.exists() and val_path.exists()):
                need_create = True
                break
        
        if need_create:
            return self._create_fold_splits(train_df, test_df)
        
        # Load existing folds
        folds = []
        for fold_idx in range(self.n_folds):
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            test_copy_path = self.splits_dir / f"fold{fold_idx}_test.csv"
            
            # Ensure test copy exists
            if not test_copy_path.exists():
                test_df.to_csv(test_copy_path, index=False)
            
            fold_train = pd.read_csv(train_path)
            fold_val = pd.read_csv(val_path)
            
            # Validate required columns
            if 'question_id' not in fold_train.columns:
                raise KeyError(f"fold{fold_idx}_train.csv missing 'question_id' column. Available: {fold_train.columns.tolist()}")
            
            folds.append((fold_train, fold_val))
        
        return folds

    def _create_fold_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create K folds from the final training set (by speaker, stratified if classification)."""
        print(f"Creating {self.n_folds} folds from final training set.")
        
        # Validate required columns
        required_cols = ["speaker_id", "label", "question_id"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Required column '{col}' not found in training data. Available: {train_df.columns.tolist()}")
        
        # Group by speaker to get labels for stratification
        speakers = train_df.groupby("speaker_id")["label"].first().reset_index()
        speakers.columns = ["speaker_id", "label"]

        if self.task == "classification":
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(kf.split(speakers, speakers["label"]))
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(kf.split(speakers))

        folds = []
        for fold_idx, (train_speaker_idx, val_speaker_idx) in enumerate(fold_splits):
            train_speakers = speakers.iloc[train_speaker_idx]["speaker_id"].values
            val_speakers = speakers.iloc[val_speaker_idx]["speaker_id"].values
            
            fold_train = train_df[train_df["speaker_id"].isin(train_speakers)].reset_index(drop=True)
            fold_val = train_df[train_df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
            
            # Double-check that question_id is preserved
            if 'question_id' not in fold_train.columns:
                raise RuntimeError(f"question_id lost when creating fold {fold_idx}")
            
            # Save train and val splits
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            fold_train.to_csv(train_path, index=False)
            fold_val.to_csv(val_path, index=False)
            
            # Save a copy of the final test set as fold*_test.csv
            test_copy_path = self.splits_dir / f"fold{fold_idx}_test.csv"
            test_df.to_csv(test_copy_path, index=False)
            
            folds.append((fold_train, fold_val))
        
        print(f"Created {self.n_folds} fold splits (train/val) and copied final test set.")
        return folds

    def _speaker_split(self, df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Split indices by speaker (stratified for classification)."""
        df_work = df.copy()
        label_col = df_work["label"]
        if isinstance(label_col, pd.DataFrame):
            label_col = label_col.iloc[:, 0]
        df_work["label"] = label_col

        speaker_labels = df_work.groupby("speaker_id")["label"].first().reset_index()
        speaker_labels.columns = ["speaker_id", "label"]

        if self.task == "classification":
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_speaker_idx, test_speaker_idx = next(
                splitter.split(speaker_labels, speaker_labels["label"])
            )
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_speaker_idx, test_speaker_idx = next(splitter.split(speaker_labels))

        train_speakers = speaker_labels.iloc[train_speaker_idx]["speaker_id"].values
        test_speakers = speaker_labels.iloc[test_speaker_idx]["speaker_id"].values
        train_idx = df_work[df_work["speaker_id"].isin(train_speakers)].index.to_numpy()
        test_idx = df_work[df_work["speaker_id"].isin(test_speakers)].index.to_numpy()
        return train_idx, test_idx


# ----------------------------------------------------------------------
#  Hyperparameter search using cross‑validation
# ----------------------------------------------------------------------
def hyperparameter_search(
    train_df: pd.DataFrame,
    split_manager: SplitManager,
    args,
    metadata: dict,
    test_df: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """
    Perform grid search over hyperparameters using K‑fold CV on the final training set.
    Returns best parameters (learning_rate, batch_size, epochs, etc.).
    """
    print("Starting hyperparameter search (nested CV) for per‑question models.")
    
    # Validate that train_df has required columns
    if 'question_id' not in train_df.columns:
        raise KeyError(f"train_df missing 'question_id' column. Available: {train_df.columns.tolist()}")
    
    # Choose a representative question (first with enough data)
    candidate_questions = [q.upper() for q in args.questions]
    rep_question = None
    for q in candidate_questions:
        q_train = train_df[train_df["question_id"] == q]
        if len(q_train) >= 20:
            rep_question = q
            break
    
    if rep_question is None:
        print("No question with enough training data for hyperparameter search. Using defaults.")
        return {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "regression_metric": getattr(args, 'regression_metric', 'rmse'),
        }

    print(f"Using question '{rep_question}' for hyperparameter search.")
    
    # Determine optimization metric and direction
    if args.task == "classification":
        metric_name = "macro_f1"
        higher_is_better = True
        print(f"Optimizing for {metric_name} (higher is better)")
    else:
        metric_name = getattr(args, 'regression_metric', 'rmse')
        higher_is_better = False  # Lower MAE/RMSE is better
        print(f"Optimizing for {metric_name} (lower is better)")
    
    # Get fold splits
    folds = split_manager.get_fold_splits(train_df, test_df)
    
    if not folds:
        print("No folds created. Using default hyperparameters.")
        return {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "regression_metric": metric_name,
        }

    # Build parameter grid from args
    param_grid = {
        "learning_rate": args.hp_learning_rates if args.hp_learning_rates else [1e-5, 2e-5, 3e-5],
        "batch_size": args.hp_batch_sizes if args.hp_batch_sizes else [8, 16],
        "epochs": args.hp_epochs if args.hp_epochs else [5, 8, 10],
        "weight_decay": args.hp_weight_decays if args.hp_weight_decays else [0.01, 0.1],
        "warmup_ratio": args.hp_warmup_ratios if args.hp_warmup_ratios else [0.06, 0.1],
        "max_length": args.hp_max_lengths if args.hp_max_lengths else [128, 256],
    }
    
    # Remove any empty lists
    param_grid = {k: v for k, v in param_grid.items() if v}
    
    # Generate parameter combinations
    keys = param_grid.keys()
    all_combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    
    # Limit combinations if specified
    if args.hp_max_combinations and len(all_combinations) > args.hp_max_combinations:
        print(f"Limiting to {args.hp_max_combinations} random combinations (from {len(all_combinations)})")
        import random as rand
        rand.seed(args.seed)
        param_combinations = rand.sample(all_combinations, args.hp_max_combinations)
    else:
        param_combinations = all_combinations
    
    print(f"Searching over {len(param_combinations)} parameter combinations")
    print(f"Parameter grid sizes: { {k: len(v) for k, v in param_grid.items()} }")
    
    best_score = -float("inf") if higher_is_better else float("inf")
    best_params = None
    best_raw_scores = []
    temp_hpo_root = Path(args.output_dir) / "temp_hpo"
    
    # For each parameter combination, evaluate on each fold and average
    for combo_idx, params in enumerate(param_combinations):
        lr = params["learning_rate"]
        bs = params["batch_size"]
        ep = params["epochs"]
        wd = params["weight_decay"]
        wr = params["warmup_ratio"]
        ml = params["max_length"]
        
        print(f"\nEvaluating combo {combo_idx+1}/{len(param_combinations)}: LR={lr}, BS={bs}, Epochs={ep}, WD={wd}, WR={wr}, ML={ml}")
        
        fold_scores = []
        fold_raw_scores = []
        
        for fold_idx, (fold_train, fold_val) in enumerate(folds):
            # Verify question_id exists in fold
            if 'question_id' not in fold_train.columns:
                print(f"ERROR: fold {fold_idx} missing question_id column")
                continue
            
            # Filter for the representative question
            q_fold_train = fold_train[fold_train["question_id"] == rep_question].reset_index(drop=True)
            q_fold_val = fold_val[fold_val["question_id"] == rep_question].reset_index(drop=True)
            
            if len(q_fold_train) == 0 or len(q_fold_val) == 0:
                continue
            
            # Create temporary config
            temp_out = Path(args.output_dir) / "temp_hpo" / rep_question / f"lr{lr}_bs{bs}_ep{ep}_wd{wd}_wr{wr}_ml{ml}_fold{fold_idx}"
            temp_cfg = TrainConfig(
                asr_file=args.asr_file,
                demo_file=args.demo_file,
                target_column=args.target_column,
                task=args.task,
                output_dir=str(temp_out),
                model_name=args.model_name,
                text_mode="question",
                aggregate_level="speaker",
                num_folds=1,
                test_size=0.0,
                final_dev_size=0.0,
                seed=args.seed,
                max_length=ml,
                batch_size=bs,
                eval_batch_size=bs,
                epochs=ep,
                learning_rate=lr,
                weight_decay=wd,
                warmup_ratio=wr,
                patience=args.patience,
                class_weights=args.class_weights,
                loss=args.loss,
                focal_gamma=args.focal_gamma,
                filter_questions=[rep_question],
                min_text_chars=args.min_text_chars,
                regression_metric=metric_name,  # Pass the metric to config
            )
            
            # Train and get validation metrics
            metrics = _train_and_evaluate(q_fold_train, q_fold_val, temp_cfg, metadata)
            
            if metrics is not None:
                raw_score = metrics[metric_name]
                fold_raw_scores.append(raw_score)
                
                # Convert score for consistent maximization
                if higher_is_better:
                    score = raw_score
                else:
                    score = -raw_score  # Negate so lower raw scores become higher scores
                
                fold_scores.append(score)
                print(f"  Fold {fold_idx}: {metric_name}={raw_score:.4f} → score={score:.4f}")
        
        if fold_scores:
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            avg_raw = np.mean(fold_raw_scores)
            
            print(f"  Average: score={avg_score:.4f} (±{std_score:.4f}), raw {metric_name}={avg_raw:.4f}")
            
            # Update best parameters
            if higher_is_better:
                is_better = avg_score > best_score
            else:
                is_better = avg_score < best_score
            
            if is_better or best_params is None:
                best_score = avg_score
                best_params = params.copy()
                best_raw_scores = fold_raw_scores
                print(f"  *** New best! {metric_name}={avg_raw:.4f} with params: {best_params} ***")
    
    # Clean up temporary directories after search
    cleanup_temp_dirs(
        Path(args.output_dir), 
        keep_best=True, 
        best_params=best_params
    )

    if best_params is None:
        print("No successful parameter combinations found. Using defaults.")
        best_params = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "regression_metric": metric_name,
        }
    else:
        print(f"\n=== Hyperparameter Search Complete ===")
        print(f"Best parameters: {best_params}")
        print(f"Best {metric_name}: {np.mean(best_raw_scores):.4f} (±{np.std(best_raw_scores):.4f})")
    
    # Save to output dir
    best_path = Path(args.output_dir) / "best_hyperparams.json"
    best_path.write_text(json.dumps(best_params, indent=2))
    
    return best_params


def _train_and_evaluate(
    train_df, 
    val_df, 
    cfg: TrainConfig, 
    metadata: dict, 
    regression_metric: str = "rmse"
) -> dict | None:
    """Train a model and return validation metrics."""
    from transformers import AutoModelForSequenceClassification
    from .model import load_tokenizer
    from .train import choose_device, train_one_fold, saved_model_exists
    from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
    import numpy as np

    model_dir = Path(cfg.output_dir) / "model"
    if not (model_dir.exists() and saved_model_exists(model_dir)):
        try:
            train_one_fold(train_df, val_df, cfg, metadata, Path(cfg.output_dir))
        except Exception as e:
            print(f"Training failed: {e}")
            return None

    # Load model and tokenizer for evaluation
    device = choose_device()
    tokenizer = load_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    texts = val_df["text"].tolist()
    labels = val_df["label"].values
    preds = []
    batch_size = cfg.eval_batch_size

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start+batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()
            
            if cfg.task == "classification":
                batch_preds = np.argmax(logits, axis=1)
                preds.extend(batch_preds)
            else:  # regression
                # For regression, logits shape is (batch_size, 1)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_preds = logits[:, 0]
                elif logits.ndim == 1:
                    batch_preds = logits
                elif logits.ndim == 0:
                    batch_preds = np.array([logits.item()])
                else:
                    batch_preds = logits.flatten()
                
                if len(batch_preds) != len(batch_texts):
                    print(f"Warning: Expected {len(batch_texts)} predictions, got {len(batch_preds)}")
                    if len(batch_preds) > len(batch_texts):
                        batch_preds = batch_preds[:len(batch_texts)]
                    else:
                        batch_preds = np.pad(batch_preds, (0, len(batch_texts) - len(batch_preds)))
                
                preds.extend(batch_preds.tolist())

    preds = np.array(preds)
    labels = np.array(labels)
    
    if len(preds) != len(labels):
        print(f"Error: Final predictions length ({len(preds)}) != labels length ({len(labels)})")
        return None
    
    if cfg.task == "classification":
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"macro_f1": macro_f1}
    else:
        mae = mean_absolute_error(labels, preds)
        rmse = np.sqrt(mean_squared_error(labels, preds))
        
        if regression_metric == "mae":
            return {"mae": mae}
        else:
            return {"rmse": rmse}
# ----------------------------------------------------------------------
#  Main ensemble pipeline
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(description="Train per‑question models with nested CV for hyperparameter tuning and top‑k selection using a fixed validation set.")
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-dir", required=True, help="Directory containing final_train/val/test.csv and fold*_train/val.csv; created if missing.")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--questions", nargs="+", default=[f"Q{i}" for i in range(1, 14)])
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction for final training set (rest split equally into val and test).")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction for final validation set.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction for final test set.")
    parser.add_argument("--n-cv-folds", type=int, default=5, help="Number of inner CV folds for hyperparameter tuning.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regression-metric", choices=["mae", "rmse"], default="rmse", help="Metric to optimize for regression tasks (default: rmse)")
    
    # Default hyperparameters (fallback if no tuning)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    
    # Hyperparameter search space
    parser.add_argument("--hp-learning-rates", nargs="+", type=float, default=None,
                        help="Learning rates to try (e.g., 1e-5 2e-5 3e-5 5e-5)")
    parser.add_argument("--hp-batch-sizes", nargs="+", type=int, default=None,
                        help="Batch sizes to try (e.g., 4 8 16 32)")
    parser.add_argument("--hp-epochs", nargs="+", type=int, default=None,
                        help="Number of epochs to try (e.g., 3 4 5 6 8 10)")
    parser.add_argument("--hp-weight-decays", nargs="+", type=float, default=None,
                        help="Weight decays to try (e.g., 0.0 0.01 0.1 0.001)")
    parser.add_argument("--hp-warmup-ratios", nargs="+", type=float, default=None,
                        help="Warmup ratios to try (e.g., 0.0 0.06 0.1 0.2)")
    parser.add_argument("--hp-max-lengths", nargs="+", type=int, default=None,
                        help="Max sequence lengths to try (e.g., 128 256 384 512)")

    # Optuna hyperparameter optimization
    parser.add_argument("--hpo-backend", choices=["grid", "random", "optuna"], default="optuna",
                        help="Hyperparameter optimization backend")
    parser.add_argument("--hpo-n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--hpo-timeout", type=int, default=None,
                        help="Timeout in seconds for Optuna search (default: no timeout)")
    parser.add_argument("--hpo-folds", type=int, default=3,
                        help="Number of CV folds for HPO (default: 3)")
    
    # Optuna advanced options
    parser.add_argument("--optuna-sampler", choices=["tpe", "random", "cmaes"], default="tpe",
                        help="Optuna sampler (default: tpe)")
    parser.add_argument("--optuna-pruner", choices=["median", "hyperband", "none"], default="median",
                        help="Optuna pruner (default: median)")

    
    # Search strategy
    parser.add_argument("--hp-random-search", action="store_true",
                        help="Use random search instead of full grid search")
    parser.add_argument("--hp-n-iterations", type=int, default=50,
                        help="Number of iterations for random search")
    parser.add_argument("--hp-max-combinations", type=int, default=None,
                        help="Maximum number of combinations to try (randomly sampled if exceeded)")
    
    # Other training parameters
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--min-text-chars", type=int, default=1)
    
    # Meta-model parameters
    parser.add_argument("--meta-model", choices=["linear", "random_forest"], default="linear")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-hpo", action="store_true", help="Force hyperparameter search even if best_hyperparams.json exists.")
    parser.add_argument("--top-k", type=int, default=0, help="If >0, evaluate all k up to that value; if 0, evaluate all.")
    
    return parser


def make_question_cfg(args, question: str, question_dir: Path, best_hparams: dict) -> TrainConfig:
    return TrainConfig(
        asr_file=args.asr_file,
        demo_file=args.demo_file,
        target_column=args.target_column,
        task=args.task,
        output_dir=str(question_dir),
        model_name=args.model_name,
        text_mode="question",
        aggregate_level="speaker",
        num_folds=1,
        test_size=0.0,
        final_dev_size=0.0,
        seed=args.seed,
        max_length=best_hparams.get("max_length", args.max_length),
        batch_size=best_hparams["batch_size"],
        eval_batch_size=best_hparams["batch_size"],
        epochs=best_hparams["epochs"],
        learning_rate=best_hparams["learning_rate"],
        weight_decay=best_hparams.get("weight_decay", args.weight_decay),
        warmup_ratio=best_hparams.get("warmup_ratio", args.warmup_ratio),
        patience=args.patience,
        class_weights=args.class_weights,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        filter_questions=[question],
        min_text_chars=args.min_text_chars,
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def extract_embeddings(model_dir: Path, df: pd.DataFrame, args, output_csv: Path, max_length: int):
    if output_csv.exists() and not args.force_embeddings:
        return pd.read_csv(output_csv)

    device = choose_device()
    tokenizer = load_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    rows = []
    texts = df["text"].tolist()
    for start in range(0, len(texts), args.embedding_batch_size):
        batch_df = df.iloc[start:start+args.embedding_batch_size].reset_index(drop=True)
        enc = tokenizer(batch_df["text"].tolist(), truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True)
        embeddings = mean_pool(outputs.hidden_states[-1], enc["attention_mask"]).cpu().numpy()

        for row_idx, embedding in enumerate(embeddings):
            meta = batch_df.iloc[row_idx]
            row = {
                "speaker_id": meta["speaker_id"],
                "session_id": meta["session_id"],
                "utterance_id": meta["utterance_id"],
                "question_id": meta["question_id"],
                "y_true": meta["label"],
            }
            row.update({f"emb_{i}": float(v) for i, v in enumerate(embedding)})
            rows.append(row)

    emb_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(output_csv, index=False)
    return emb_df


def train_question_models(train_df, val_df, test_df, metadata, args, best_hparams, out_dir: Path):
    embedding_files = {"train": {}, "val": {}, "test": {}}
    summaries = []

    for question in [q.upper() for q in args.questions]:
        q_train = train_df[train_df["question_id"] == question].reset_index(drop=True)
        q_val = val_df[val_df["question_id"] == question].reset_index(drop=True)
        q_test = test_df[test_df["question_id"] == question].reset_index(drop=True)

        if q_train.empty:
            print(f"{question}: skipping, no training examples")
            continue

        q_dir = out_dir / "question_models" / question
        model_dir = q_dir / "model"
        train_emb = q_dir / "embeddings_train.csv"
        val_emb = q_dir / "embeddings_val.csv"
        test_emb = q_dir / "embeddings_test.csv"

        if (model_dir.exists() and saved_model_exists(model_dir) and
            train_emb.exists() and val_emb.exists() and (q_test.empty or test_emb.exists())):
            print(f"{question}: model and embeddings already exist, loading.")
            embedding_files["train"][question] = train_emb
            embedding_files["val"][question] = val_emb
            embedding_files["test"][question] = test_emb if not q_test.empty else None
            summaries.append({"question_id": question, "train_examples": len(q_train),
                              "val_examples": len(q_val), "test_examples": len(q_test),
                              "model_dir": str(model_dir)})
            continue

        print(f"{question}: training model on {len(q_train)} examples, val on {len(q_val)}")
        q_cfg = make_question_cfg(args, question, q_dir, best_hparams)
        train_one_fold(q_train, q_val, q_cfg, metadata, q_dir)
        if not saved_model_exists(model_dir):
            raise FileNotFoundError(f"Expected saved model at {model_dir}")

        extract_embeddings(model_dir, q_train, args, train_emb, best_hparams["max_length"])
        extract_embeddings(model_dir, q_val, args, val_emb, best_hparams["max_length"])
        if not q_test.empty:
            extract_embeddings(model_dir, q_test, args, test_emb, best_hparams["max_length"])

        embedding_files["train"][question] = train_emb
        embedding_files["val"][question] = val_emb
        embedding_files["test"][question] = test_emb if not q_test.empty else None
        summaries.append({"question_id": question, "train_examples": len(q_train),
                          "val_examples": len(q_val), "test_examples": len(q_test),
                          "model_dir": str(model_dir)})

    pd.DataFrame(summaries).to_csv(out_dir / "question_model_summary.csv", index=False)
    return embedding_files


def build_feature_table(embedding_paths: dict[str, Path | None], questions: list[str]):
    tables = []
    for q in questions:
        path = embedding_paths.get(q)
        if path is None or not Path(path).exists():
            continue
        emb_df = pd.read_csv(path)
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        if not emb_cols:
            continue
        grouped = emb_df.groupby("speaker_id", as_index=True).agg(
            y_true=("y_true", "first"),
            **{col: (col, "mean") for col in emb_cols},
        )
        grouped = grouped.rename(columns={col: f"{q}__{col}" for col in emb_cols})
        grouped[f"{q}__present"] = 1.0
        tables.append(grouped)
    if not tables:
        raise ValueError("No embedding tables available.")
    merged = tables[0]
    for t in tables[1:]:
        merged = merged.join(t.drop(columns=["y_true"]), how="outer")
        merged["y_true"] = merged["y_true"].combine_first(t["y_true"])
    merged = merged.reset_index()
    feature_cols = [c for c in merged.columns if "__" in c]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    return merged, feature_cols


def align_feature_tables(train_df, val_df, test_df, feature_cols):
    for df in (val_df, test_df):
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        extra = [c for c in df.columns if "__" in c and c not in feature_cols]
        if extra:
            df.drop(columns=extra, inplace=True)
    return train_df, val_df, test_df


def make_meta_model(args):
    if args.task == "classification":
        if args.meta_model == "random_forest":
            return RandomForestClassifier(
                n_estimators=args.n_estimators, random_state=args.seed,
                class_weight="balanced", min_samples_leaf=2, n_jobs=-1
            )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=args.seed)),
        ])
    if args.meta_model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators, random_state=args.seed,
            min_samples_leaf=2, n_jobs=-1
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def score_meta_model(model, x, y, task):
    pred = model.predict(x)
    if task == "classification":
        return {
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": r2_score(y, pred)}


def primary_score(metrics: dict, task: str) -> float:
    return metrics["macro_f1"] if task == "classification" else -metrics["mae"]


def question_groups(feature_cols):
    groups = {}
    for c in feature_cols:
        q = c.split("__", 1)[0]
        groups.setdefault(q, []).append(c)
    return groups


def permutation_question_importance(model, val_df, feature_cols, args):
    x_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y_true"].to_numpy()
    base_metrics = score_meta_model(model, x_val, y_val, args.task)
    base_score = primary_score(base_metrics, args.task)
    groups = question_groups(feature_cols)
    rng = np.random.RandomState(args.seed)
    rows = []
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    for q, cols in groups.items():
        indices = [col_to_idx[c] for c in cols]
        drops = []
        for _ in range(args.permutation_repeats):
            x_perm = x_val.copy()
            shuffled = x_perm[:, indices].copy()
            rng.shuffle(shuffled)
            x_perm[:, indices] = shuffled
            m = score_meta_model(model, x_perm, y_val, args.task)
            drops.append(base_score - primary_score(m, args.task))
        rows.append({"question_id": q, "importance": float(np.mean(drops)), "importance_std": float(np.std(drops)), "base_score": float(base_score)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def train_meta_model(train_features, val_features, test_features, feature_cols, args, out_dir: Path):
    # Compute question importance using the validation set
    base_model = make_meta_model(args)
    base_model.fit(train_features[feature_cols].to_numpy(), train_features["y_true"].to_numpy())
    importance_df = permutation_question_importance(base_model, val_features, feature_cols, args)
    importance_df.to_csv(out_dir / "question_embedding_importance.csv", index=False)

    questions_ranked = importance_df["question_id"].tolist()
    if not questions_ranked:
        raise ValueError("No ranked questions.")

    max_k = len(questions_ranked)
    ks = list(range(1, max_k + 1))
    if args.top_k and 0 < args.top_k < max_k:
        ks = sorted(set(ks + [args.top_k]))

    # Evaluate each k on validation set
    val_metrics = {}
    best_val_score = -float("inf") if args.task == "classification" else float("inf")
    best_k = 1
    for k in ks:
        selected_qs = questions_ranked[:k]
        selected_cols = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs)]
        model = make_meta_model(args)
        model.fit(train_features[selected_cols].to_numpy(), train_features["y_true"].to_numpy())
        m = score_meta_model(model, val_features[selected_cols].to_numpy(), val_features["y_true"].to_numpy(), args.task)
        val_metrics[k] = m
        score = m["macro_f1"] if args.task == "classification" else -m["mae"]
        if (args.task == "classification" and score > best_val_score) or (args.task == "regression" and score < best_val_score):
            best_val_score = score
            best_k = k
    print(f"Best k on validation set: {best_k} (score: {best_val_score})")

    # Retrain on train+val with best_k
    trainval_features = pd.concat([train_features, val_features], ignore_index=True)
    selected_qs_final = questions_ranked[:best_k]
    selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
    final_model = make_meta_model(args)
    final_model.fit(trainval_features[selected_cols_final].to_numpy(), trainval_features["y_true"].to_numpy())

    # Final evaluation on test set
    test_metrics = score_meta_model(
        final_model,
        test_features[selected_cols_final].to_numpy(),
        test_features["y_true"].to_numpy(),
        args.task
    )
    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_dir / "meta_model.joblib")
    pd.DataFrame({"question_id": selected_qs_final}).to_csv(out_dir / "selected_questions.csv", index=False)
    pd.DataFrame({"feature": selected_cols_final}).to_csv(out_dir / "selected_embedding_features.csv", index=False)
    with open(out_dir / "meta_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Predictions
    preds = final_model.predict(test_features[selected_cols_final].to_numpy())
    out_df = test_features[["speaker_id", "y_true"]].copy()
    out_df["y_pred"] = preds
    if args.task == "classification" and hasattr(final_model, "predict_proba"):
        probs = final_model.predict_proba(test_features[selected_cols_final].to_numpy())
        classes = final_model.classes_
        for i, cls in enumerate(classes):
            out_df[f"prob_{cls}"] = probs[:, i]
    out_df.to_csv(out_dir / "meta_test_predictions.csv", index=False)

    # Save validation metrics per k
    val_summary = []
    for k, m in val_metrics.items():
        row = {"top_k": k, "questions": ",".join(questions_ranked[:k])}
        row.update({k2: v2 for k2, v2 in m.items() if isinstance(v2, (int, float, str))})
        val_summary.append(row)
    pd.DataFrame(val_summary).to_csv(out_dir / "topk_val_metrics.csv", index=False)

    return {"best_k": best_k, "best_val_score": best_val_score, "test_metrics": test_metrics}


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    # Save config
    (out_dir / "question_ensemble_config.json").write_text(json.dumps(vars(args), indent=2))

    # Load full dataset
    questions = [q.upper() for q in args.questions]
    df, metadata = load_examples(
        args.asr_file, args.demo_file, args.target_column, args.task,
        text_mode="question", min_text_chars=args.min_text_chars,
        filter_questions=questions,
    )
    
    # Verify question_id exists
    if 'question_id' not in df.columns:
        raise ValueError(f"Loaded DataFrame missing 'question_id' column. Available: {df.columns.tolist()}")
    
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Manage splits
    split_mgr = SplitManager(
        splits_dir, args.task,
        args.train_frac, args.val_frac, args.test_frac,
        args.seed, args.n_cv_folds
    )
    final_train, final_val, final_test = split_mgr.get_final_splits(df)
    print(f"Final splits: train={len(final_train)}, val={len(final_val)}, test={len(final_test)}")
    
    # Verify question_id in splits
    for name, split_df in [("train", final_train), ("val", final_val), ("test", final_test)]:
        if 'question_id' not in split_df.columns:
            raise KeyError(f"final_{name} missing 'question_id' column")

    # Hyperparameter search
    best_hparams_path = out_dir / "best_hyperparams.json"
    if best_hparams_path.exists() and not args.force_hpo:
        best_hparams = json.loads(best_hparams_path.read_text())
        print(f"Loaded best hyperparameters from {best_hparams_path}: {best_hparams}")
    else:
        if args.hpo_backend == "optuna":
            best_hparams = hyperparameter_search_optuna(
                final_train, split_mgr, args, metadata, final_test, args.n_cv_folds
            )
        elif args.hpo_backend == "random":
            # Use existing random search from previous code
            args.hp_random_search = True
            args.hp_n_iterations = args.hpo_n_trials
            best_hparams = hyperparameter_search(
                final_train, split_mgr, args, metadata, final_test, args.n_cv_folds
            )
        else:  # grid search
            best_hparams = hyperparameter_search(
                final_train, split_mgr, args, metadata, final_test, args.n_cv_folds
            )

    if args.task == "regression" and 'regression_metric' not in best_hparams:
        best_hparams['regression_metric'] = getattr(args, 'regression_metric', 'rmse')
    
    # Update args for later use
    args.learning_rate = best_hparams["learning_rate"]
    args.batch_size = best_hparams["batch_size"]
    args.epochs = best_hparams["epochs"]
    args.weight_decay = best_hparams.get("weight_decay", args.weight_decay)
    args.warmup_ratio = best_hparams.get("warmup_ratio", args.warmup_ratio)
    args.max_length = best_hparams.get("max_length", args.max_length)

    # Train per‑question models using best hyperparameters
    embedding_files = train_question_models(final_train, final_val, final_test, metadata, args, best_hparams, out_dir)

    # Build feature tables
    available_qs = list(embedding_files["train"].keys())
    train_features, feature_cols = build_feature_table(embedding_files["train"], available_qs)
    val_features, _ = build_feature_table(embedding_files["val"], available_qs)
    test_features, _ = build_feature_table(embedding_files["test"], available_qs)
    train_features, val_features, test_features = align_feature_tables(train_features, val_features, test_features, feature_cols)

    # Save raw feature tables
    train_features.to_csv(out_dir / "meta_train_features.csv", index=False)
    val_features.to_csv(out_dir / "meta_val_features.csv", index=False)
    test_features.to_csv(out_dir / "meta_test_features.csv", index=False)

    # Train meta‑model
    results = train_meta_model(train_features, val_features, test_features, feature_cols, args, out_dir)
    print("\n===== Final Results =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()