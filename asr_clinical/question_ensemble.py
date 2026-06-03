from __future__ import annotations

import shap
import lightgbm as lgb 
import argparse
import json
import random
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
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
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from transformers import AutoModelForSequenceClassification

from .config import TrainConfig
from .data import load_examples
from .model import load_tokenizer
from .train import choose_device, saved_model_exists, train_one_fold
import shutil

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# Also add a cache directory to avoid redownloading
import os
os.environ['HF_HOME'] = '/home/bahman/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/bahman/.cache/huggingface/transformers'

# ----------------------------------------------------------------------
#  Utility Functions
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_hist_gradient_boosting(task, args):
    """Native scikit-learn HistGradientBoosting – no version conflicts."""
    if task == "classification":
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            learning_rate=args.xgb_lr,
            max_depth=getattr(args, 'max_depth', None),
            random_state=args.seed,
            verbose=0
        )
    else:  # regression
        return HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.xgb_lr,
            max_depth=getattr(args, 'max_depth', None),
            random_state=args.seed,
            verbose=0
        )

def cleanup_old_splits(splits_dir: Path):
    """Delete existing split files to force regeneration with correct columns."""
    if splits_dir.exists():
        print(f"Checking for old split files in {splits_dir}")
        deleted = False
        for pattern in ["fold*_train.csv", "fold*_val.csv", "fold*_test.csv", "final_*.csv"]:
            for f in splits_dir.glob(pattern):
                print(f"  Removing old file: {f.name}")
                f.unlink()
                deleted = True
        if deleted:
            print("  Old split files removed. Will regenerate with correct columns.")
        else:
            print("  No existing split files found.")


def cleanup_temp_dirs(temp_dir: Path):
    """Clean up temporary directories created during hyperparameter search."""
    if not temp_dir.exists():
        return
    
    temp_hpo_dir = temp_dir / "temp_hpo"
    if temp_hpo_dir.exists():
        print(f"\nCleaning up temporary hyperparameter search directories...")
        shutil.rmtree(temp_hpo_dir)
        print(f"  Removed {temp_hpo_dir}")
    
    temp_optuna_dir = temp_dir / "temp_hpo_optuna"
    if temp_optuna_dir.exists():
        shutil.rmtree(temp_optuna_dir)
        print(f"  Removed {temp_optuna_dir}")


# ----------------------------------------------------------------------
#  Split Management
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
        self._validate_or_cleanup_splits()

    def _validate_or_cleanup_splits(self):
        """Check if existing splits have required columns, clean up if not."""
        required_cols = ['question_id', 'label', 'speaker_id']
        
        final_train = self.splits_dir / "final_train.csv"
        if final_train.exists():
            try:
                sample = pd.read_csv(final_train, nrows=1)
                missing = [col for col in required_cols if col not in sample.columns]
                if missing:
                    print(f"Existing final splits missing columns: {missing}. Deleting and regenerating...")
                    self._delete_all_splits()
                    return
            except Exception:
                self._delete_all_splits()
                return
        
        for fold_idx in range(self.n_folds):
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            if train_path.exists():
                try:
                    sample = pd.read_csv(train_path, nrows=1)
                    missing = [col for col in required_cols if col not in sample.columns]
                    if missing:
                        print(f"Existing fold splits missing columns: {missing}. Deleting and regenerating...")
                        self._delete_all_splits()
                        return
                except Exception:
                    self._delete_all_splits()
                    return

    def _delete_all_splits(self):
        """Delete all split files."""
        for pattern in ["fold*_train.csv", "fold*_val.csv", "fold*_test.csv", "final_*.csv"]:
            for f in self.splits_dir.glob(pattern):
                f.unlink()

    def get_final_splits(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load or create final_train/val/test CSV files."""
        train_path = self.splits_dir / "final_train.csv"
        val_path = self.splits_dir / "final_val.csv"
        test_path = self.splits_dir / "final_test.csv"

        if train_path.exists() and val_path.exists() and test_path.exists():
            print("Loading existing final splits.")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            
            for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                if 'question_id' not in split_df.columns:
                    raise KeyError(f"final_{name}.csv missing 'question_id' column.")
            
            return train_df, val_df, test_df

        print("Creating final train/val/test splits (by speaker).")
        
        # Handle test_frac == 0 case
        if self.test_frac == 0:
            print("test_frac=0: No test set will be created.")
            # Split only into train and val
            rel_val_frac = self.val_frac / (self.train_frac + self.val_frac)
            train_idx, val_idx = self._speaker_split(df, rel_val_frac, self.seed)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = pd.DataFrame()  # Empty test DataFrame
            
            # Save splits
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            # Create empty test file with same columns
            if len(df) > 0:
                empty_test = pd.DataFrame(columns=df.columns)
                empty_test.to_csv(test_path, index=False)
            
            print("Final splits saved (train/val only).")
            return train_df, val_df, test_df
        
        # Normal case with test_frac > 0
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
        need_create = False
        for fold_idx in range(self.n_folds):
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            if not (train_path.exists() and val_path.exists()):
                need_create = True
                break
            
            if train_path.exists():
                sample = pd.read_csv(train_path, nrows=1)
                if 'question_id' not in sample.columns or 'label' not in sample.columns:
                    print(f"Fold {fold_idx} missing required columns. Regenerating all folds.")
                    need_create = True
                    break
        
        if need_create:
            return self._create_fold_splits(train_df, test_df)
        
        folds = []
        for fold_idx in range(self.n_folds):
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            test_copy_path = self.splits_dir / f"fold{fold_idx}_test.csv"
            
            if not test_copy_path.exists() and not test_df.empty:
                test_df.to_csv(test_copy_path, index=False)
            
            fold_train = pd.read_csv(train_path)
            fold_val = pd.read_csv(val_path)
            
            if 'question_id' not in fold_train.columns:
                raise KeyError(f"fold{fold_idx}_train.csv missing 'question_id' column.")
            
            folds.append((fold_train, fold_val))
        
        return folds

    def _create_fold_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create K folds from the final training set."""
        print(f"Creating {self.n_folds} folds from final training set.")
        
        required_cols = ["speaker_id", "label", "question_id"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")
        
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
            
            if 'question_id' not in fold_train.columns:
                raise RuntimeError(f"question_id lost when creating fold {fold_idx}")
            
            train_path = self.splits_dir / f"fold{fold_idx}_train.csv"
            val_path = self.splits_dir / f"fold{fold_idx}_val.csv"
            fold_train.to_csv(train_path, index=False)
            fold_val.to_csv(val_path, index=False)
            
            if not test_df.empty:
                test_copy_path = self.splits_dir / f"fold{fold_idx}_test.csv"
                test_df.to_csv(test_copy_path, index=False)
            
            folds.append((fold_train, fold_val))
        
        print(f"Created {self.n_folds} fold splits.")
        return folds

    def _speaker_split(self, df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Split indices by speaker."""
        # Handle edge cases
        if test_size <= 0:
            # Return all indices as train, empty as test
            return np.arange(len(df)), np.array([], dtype=int)
        
        if test_size >= 1:
            # Return empty as train, all as test
            return np.array([], dtype=int), np.arange(len(df))
        
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
#  Primary Score Function
# ----------------------------------------------------------------------
def primary_score(metrics: dict, task: str) -> float:
    if task == "classification":
        return metrics.get("macro_f1", 0.0)
    else:
        return -metrics.get("rmse", float('inf'))


# ----------------------------------------------------------------------
#  Hyperparameter Search on ALL Questions
# ----------------------------------------------------------------------
def hyperparameter_search_optuna_all_questions(
    train_df: pd.DataFrame,
    split_manager: SplitManager,
    args,
    metadata: dict,
    test_df: pd.DataFrame,
) -> dict:
    """Perform hyperparameter search across ALL questions."""
    print("=" * 60)
    print("Starting Optuna hyperparameter search on ALL QUESTIONS")
    print("=" * 60)
    
    folds = split_manager.get_fold_splits(train_df, test_df)
    folds = folds[:args.hpo_folds]
    all_questions = [q.upper() for q in args.questions]
    
    print(f"Optimizing across {len(all_questions)} questions with {len(folds)}-fold CV")
    print(f"Total trials: {args.hpo_n_trials}")
    
    sampler = TPESampler(seed=args.seed, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{args.task}_hpo_all_questions",
        load_if_exists=True
    )
    
    objective_partial = partial(
        objective_function_all_questions,
        folds=folds,
        all_questions=all_questions,
        args=args,
        metadata=metadata,
    )
    
    print(f"\nRunning Optuna for {args.hpo_n_trials} trials...")
    study.optimize(
        objective_partial,
        n_trials=args.hpo_n_trials,
        timeout=args.hpo_timeout,
        show_progress_bar=True,
        n_jobs=1
    )
    
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== Optuna Search Complete ===")
    print(f"Best primary score: {best_value:.4f}")
    print(f"Best parameters: {best_params}")
    
    study_path = Path(args.output_dir) / "optuna_study_all_questions.pkl"
    joblib.dump(study, study_path)
    
    best_params.update({
        "max_length": best_params.get("max_length", args.max_length),
        "weight_decay": best_params.get("weight_decay", args.weight_decay),
        "warmup_ratio": best_params.get("warmup_ratio", args.warmup_ratio),
    })
    
    return best_params


def objective_function_all_questions(
    trial: optuna.Trial,
    folds: list,
    all_questions: list,
    args,
    metadata: dict,
) -> float:
    """Objective function that averages performance across ALL questions."""
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),  # Narrowed range
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),  # Smaller batch sizes
        "epochs": trial.suggest_int("epochs", 1, 3),  # Fewer epochs for HPO
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "max_length": trial.suggest_categorical("max_length", [128, 256]),  # Reduced options
    }
    
    print(f"\nTrial {trial.number}: testing {params}")
    
    all_question_scores = []
    
    for fold_idx, (fold_train, fold_val) in enumerate(folds):
        fold_question_scores = []
        
        for question in all_questions:
            q_fold_train = fold_train[fold_train["question_id"] == question].reset_index(drop=True)
            q_fold_val = fold_val[fold_val["question_id"] == question].reset_index(drop=True)
            
            # Need more samples for meaningful training
            if len(q_fold_train) < 10 or len(q_fold_val) < 3:
                print(f"  Skipping {question}: train={len(q_fold_train)}, val={len(q_fold_val)} (insufficient data)")
                continue
            
            temp_out = Path(args.output_dir) / "temp_hpo_optuna" / f"trial{trial.number}_fold{fold_idx}_{question}"
            
            # Add retry logic for rate limiting
            max_retries = 3
            metrics = None
            for retry in range(max_retries):
                try:
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
                        seed=args.seed + trial.number + fold_idx + retry,
                        max_length=params["max_length"],
                        batch_size=params["batch_size"],
                        eval_batch_size=params["batch_size"],
                        epochs=params["epochs"],
                        learning_rate=params["learning_rate"],
                        weight_decay=params["weight_decay"],
                        warmup_ratio=params["warmup_ratio"],
                        patience=1,  # Reduced patience for HPO
                        class_weights=args.class_weights,
                        loss=args.loss,
                        focal_gamma=args.focal_gamma,
                        filter_questions=[question],
                        min_text_chars=args.min_text_chars,
                    )
                    
                    metrics = _train_and_evaluate_fast(q_fold_train, q_fold_val, temp_cfg, metadata)
                    if metrics is not None:
                        break
                        
                except Exception as e:
                    print(f"  Attempt {retry+1} failed for {question}: {e}")
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        import time
                        wait_time = (retry + 1) * 5  # 5, 10, 15 seconds
                        print(f"  Rate limited! Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    continue
                finally:
                    # Clean up temp directory
                    try:
                        shutil.rmtree(temp_out)
                    except:
                        pass
            
            if metrics is not None:
                score = primary_score(metrics, args.task)
                fold_question_scores.append(score)
                print(f"  {question}: score={score:.4f}")
            else:
                print(f"  {question}: FAILED to train")
        
        if fold_question_scores:
            fold_avg_score = np.mean(fold_question_scores)
            all_question_scores.append(fold_avg_score)
            print(f"Fold {fold_idx} average score: {fold_avg_score:.4f}")
            trial.report(np.mean(all_question_scores), fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    if not all_question_scores:
        print(f"Trial {trial.number}: No valid scores - returning -inf")
        return float('-inf')
    
    final_score = np.mean(all_question_scores)
    print(f"Trial {trial.number} final score: {final_score:.4f}")
    return final_score

def download_with_retry(model_name: str, max_retries: int = 5):
    """Download model with retry logic for rate limiting."""
    from transformers import AutoConfig
    import time
    
    for retry in range(max_retries):
        try:
            # Try to load the config first
            config = AutoConfig.from_pretrained(model_name)
            return config
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (retry + 1) * 10  # 10, 20, 30, 40, 50 seconds
                print(f"Rate limited! Waiting {wait_time} seconds before retry {retry+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(f"Failed to download {model_name} after {max_retries} retries")



def _train_and_evaluate_fast(train_df, val_df, cfg: TrainConfig, metadata: dict) -> dict | None:
    """Fast training and evaluation for hyperparameter search with caching."""
    from transformers import AutoModelForSequenceClassification
    from .model import load_tokenizer
    from .train import choose_device, train_one_fold, saved_model_exists
    
    # Create a cache key based on the data and config
    cache_key = f"{hash(frozenset(train_df['utterance_id']))}_{cfg.learning_rate}_{cfg.batch_size}_{cfg.epochs}"
    cache_path = Path(cfg.output_dir) / f"cache_{cache_key}.pkl"
    
    # Check cache
    if cache_path.exists():
        try:
            return joblib.load(cache_path)
        except:
            pass
    
    model_dir = Path(cfg.output_dir) / "model"
    if not (model_dir.exists() and saved_model_exists(model_dir)):
        try:
            train_one_fold(train_df, val_df, cfg, metadata, Path(cfg.output_dir))
        except Exception as e:
            print(f"Training failed: {e}")
            return None

    try:
        device = choose_device()
        tokenizer = load_tokenizer(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        texts = val_df["text"].tolist()
        labels = val_df["label"].values
        preds = []
        batch_size = min(cfg.eval_batch_size, len(texts))  # Ensure batch size doesn't exceed data size

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
                else:
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        batch_preds = logits[:, 0]
                    else:
                        batch_preds = logits.flatten()
                    preds.extend(batch_preds.tolist())

        preds = np.array(preds)
        labels = np.array(labels)
        
        if len(preds) != len(labels):
            return None
        
        if cfg.task == "classification":
            result = {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
        else:
            result = {"rmse": np.sqrt(mean_squared_error(labels, preds))}
        
        # Save to cache
        joblib.dump(result, cache_path)
        return result
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


# ----------------------------------------------------------------------
#  Individual Meta-Model Creators
# ----------------------------------------------------------------------
def create_linear_model(task, args):
    """Create linear model: LogisticRegression for classification, Ridge for regression"""
    if task == "classification":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000, 
                class_weight="balanced", 
                random_state=args.seed,
                C=getattr(args, 'logreg_C', 1.0)
            )),
        ])
    else:  # regression
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=getattr(args, 'ridge_alpha', 1.0))),
        ])

def create_ridge(task, args):
    """Ridge regression (regression only) - falls back to linear for classification"""
    if task == "regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=getattr(args, 'ridge_alpha', 1.0))),
        ])
    else:
        print("  Note: Ridge is for regression only, using Logistic Regression for classification")
        return create_linear_model(task, args)

def create_lasso(task, args):
    """Lasso regression (regression only) - falls back to linear for classification"""
    if task == "regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=getattr(args, 'lasso_alpha', 1.0),
                random_state=args.seed, 
                max_iter=5000
            )),
        ])
    else:
        print("  Note: Lasso is for regression only, using Logistic Regression for classification")
        return create_linear_model(task, args)

def create_elasticnet(task, args):
    """ElasticNet regression (regression only) - falls back to linear for classification"""
    if task == "regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=getattr(args, 'elasticnet_alpha', 1.0),
                l1_ratio=getattr(args, 'elasticnet_l1_ratio', 0.5),
                random_state=args.seed, 
                max_iter=5000
            )),
        ])
    else:
        print("  Note: ElasticNet is for regression only, using Logistic Regression for classification")
        return create_linear_model(task, args)

def create_random_forest(task, args):
    """Create Random Forest model (classification or regression)"""
    if task == "classification":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            max_depth=getattr(args, 'max_depth', None)
        )
    else:  # regression
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            min_samples_leaf=2,
            n_jobs=-1,
            max_depth=getattr(args, 'max_depth', None)
        )

def create_svm(task, args):
    """Create SVM model (SVC for classification, SVR for regression)"""
    if task == "classification":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel=getattr(args, 'svm_kernel', 'rbf'),
                C=getattr(args, 'svm_C', 1.0),
                gamma=getattr(args, 'svm_gamma', 'scale'),
                probability=True,  # Required for soft voting
                class_weight="balanced",
                random_state=args.seed
            )),
        ])
    else:  # regression
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel=getattr(args, 'svm_kernel', 'rbf'),
                C=getattr(args, 'svm_C', 1.0),
                epsilon=getattr(args, 'svm_epsilon', 0.1)
            )),
        ])

def create_gradient_boosting(task, args):
    """Create Gradient Boosting model (classification or regression)"""
    if task == "classification":
        return GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=0.1,
            max_depth=3,
            random_state=args.seed
        )
    else:  # regression
        return GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=0.1,
            max_depth=3,
            random_state=args.seed
        )

def create_knn(task, args):
    """Create KNN model (classification or regression)"""
    if task == "classification":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=getattr(args, 'knn_neighbors', 5),
                weights='distance'
            )),
        ])
    else:  # regression
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=getattr(args, 'knn_neighbors', 5),
                weights='distance'
            )),
        ])

def create_ensemble_model(task, args):
    """
    Create an ensemble of multiple meta-models with voting/averaging.
    
    For classification: Uses soft voting (averages probabilities)
    For regression: Uses averaging (averages predictions)
    
    Args:
        task: "classification" or "regression"
        args: Command line arguments with ensemble configuration
        
    Returns:
        VotingClassifier or VotingRegressor ensemble
    """
    
    # Get list of models to include in ensemble
    ensemble_models = getattr(args, 'ensemble_models', ['linear', 'random_forest', 'xgboost'])
    
    # Validate that models are appropriate for the task
    classification_only_models = []  # Models that only work for classification
    regression_only_models = ['ridge', 'lasso', 'elasticnet']  # Models that only work for regression
    
    if task == "classification":
        # Remove regression-only models from ensemble
        invalid_models = [m for m in ensemble_models if m in regression_only_models]
        if invalid_models:
            print(f"Warning: Removing regression-only models from classification ensemble: {invalid_models}")
            ensemble_models = [m for m in ensemble_models if m not in regression_only_models]
    else:  # regression
        # All models should have regression implementations
        # (classification_only_models is empty, so no filtering needed)
        pass
    
    if not ensemble_models:
        print("Error: No valid models for ensemble. Falling back to linear model.")
        return create_linear_model(task, args)
    
    # Map model names to creator functions
    model_creators = {
        'linear': create_linear_model,
        'ridge': create_ridge,
        'lasso': create_lasso,
        'elasticnet': create_elasticnet,
        'random_forest': create_random_forest,
        'svm': create_svm, 
        'gradient_boosting': create_gradient_boosting,
        'hist_gradient_boosting': create_hist_gradient_boosting,
        'knn': create_knn,
    }
    
    # Create individual models
    estimators = []
    failed_models = []
    
    print(f"\nCreating ensemble for {task} task with models: {ensemble_models}")
    
    for model_name in ensemble_models:
        if model_name not in model_creators:
            print(f"  ✗ Unknown model '{model_name}', skipping")
            failed_models.append(model_name)
            continue
        
        try:
            # Create the model
            model = model_creators[model_name](task, args)
            
            # Validate the model type matches the task
            if task == "classification":
                from sklearn.base import ClassifierMixin
                if not isinstance(model, ClassifierMixin):
                    # Try to check if it's a pipeline with a classifier
                    if hasattr(model, 'named_steps'):
                        last_step = list(model.named_steps.values())[-1]
                        if not isinstance(last_step, ClassifierMixin):
                            raise ValueError(f"{model_name} is not a classifier (got {type(last_step).__name__})")
                    else:
                        raise ValueError(f"{model_name} is not a classifier (got {type(model).__name__})")
            else:  # regression
                from sklearn.base import RegressorMixin
                if not isinstance(model, RegressorMixin):
                    # Try to check if it's a pipeline with a regressor
                    if hasattr(model, 'named_steps'):
                        last_step = list(model.named_steps.values())[-1]
                        if not isinstance(last_step, RegressorMixin):
                            raise ValueError(f"{model_name} is not a regressor (got {type(last_step).__name__})")
                    else:
                        raise ValueError(f"{model_name} is not a regressor (got {type(model).__name__})")
            
            estimators.append((model_name, model))
            print(f"  ✓ Added {model_name} to ensemble")
            
        except Exception as e:
            print(f"  ✗ Failed to create {model_name}: {e}")
            failed_models.append(model_name)
    
    # Remove failed models from ensemble_models list for reporting
    successful_models = [m for m in ensemble_models if m not in failed_models]
    
    if not estimators:
        print("\nNo valid ensemble models. Falling back to linear model.")
        return create_linear_model(task, args)
    
    # Create the appropriate ensemble based on task
    if task == "classification":
        # For classification: use soft voting (probability averaging)
        voting_type = getattr(args, 'ensemble_voting', 'soft')
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting_type,  # 'soft' for probability averaging, 'hard' for majority vote
            weights=getattr(args, 'ensemble_weights', None),
            n_jobs=-1
        )
        
        print(f"\n✓ Created CLASSIFICATION ensemble with {len(estimators)} models")
        print(f"  - Voting type: {voting_type} {'(probability averaging)' if voting_type == 'soft' else '(majority vote)'}")
        print(f"  - Models: {successful_models}")
        
        if getattr(args, 'ensemble_weights', None):
            print(f"  - Weights: {args.ensemble_weights}")
        
    else:  # regression
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=getattr(args, 'ensemble_weights', None),
            n_jobs=-1
        )
        
        print(f"\n✓ Created REGRESSION ensemble with {len(estimators)} models")
        print(f"  - Averaging type: weighted {'with weights' if getattr(args, 'ensemble_weights', None) else 'equal'}")
        print(f"  - Models: {successful_models}")
        
        if getattr(args, 'ensemble_weights', None):
            print(f"  - Weights: {args.ensemble_weights}")
    
    # Save ensemble information to args for later use
    args.ensemble_models_used = successful_models
    
    return ensemble

def create_ensemble_model_with_list(task, args, model_list):
    """Helper function to create ensemble with a specific list of models."""
    model_creators = {
        'linear': create_linear_model,
        'ridge': create_ridge,
        'lasso': create_lasso,
        'elasticnet': create_elasticnet,
        'random_forest': create_random_forest,
        'svm': create_svm,
        'gradient_boosting': create_gradient_boosting,
        'hist_gradient_boosting': create_hist_gradient_boosting,
        'knn': create_knn,
    }
    
    estimators = []
    for model_name in model_list:
        if model_name in model_creators:
            model = model_creators[model_name](task, args)
            estimators.append((model_name, model))
            print(f"  - Adding {model_name} to ensemble")
    
    if task == "classification":
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=getattr(args, 'ensemble_weights', None),
            n_jobs=-1
        )
    else:
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=getattr(args, 'ensemble_weights', None)
        )
    
    return ensemble


def make_meta_model(args):
    """
    Create meta-model (single or ensemble based on configuration).
    
    For single models, supports:
    - linear: Logistic Regression (classification) or Ridge (regression)
    - ridge: Ridge regression (regression only, falls back to linear for classification)
    - lasso: Lasso regression (regression only, falls back to linear for classification)
    - elasticnet: ElasticNet regression (regression only, falls back to linear for classification)
    - random_forest: Random Forest (classification or regression)
    - svm: SVM (classification or regression)
    - hist_gradient_boosting: Hist Gradient Boosting (classification or regression)
    - gradient_boosting: Gradient Boosting (classification or regression)
    - knn: KNN (classification or regression)
    
    For ensembles, use --use-ensemble and --ensemble-models
    """
    
    if getattr(args, 'use_ensemble', False):
        print("\n" + "=" * 50)
        print("CREATING ENSEMBLE META-MODEL")
        print("=" * 50)
        return create_ensemble_model(args.task, args)
    else:
        # Single model (original behavior)
        print(f"\nCreating single meta-model: {args.meta_model}")
        
        if args.meta_model == "linear":
            return create_linear_model(args.task, args)
        elif args.meta_model == "ridge":
            return create_ridge(args.task, args)
        elif args.meta_model == "lasso":
            return create_lasso(args.task, args)
        elif args.meta_model == "elasticnet":
            return create_elasticnet(args.task, args)
        elif args.meta_model == "random_forest":
            return create_random_forest(args.task, args)
        elif args.meta_model == "svm":
            return create_svm(args.task, args)
        elif args.meta_model == "hist_gradient_boosting":
            return create_hist_gradient_boosting(args.task, args)
        elif args.meta_model == "gradient_boosting":
            return create_gradient_boosting(args.task, args)
        elif args.meta_model == "knn":
            return create_knn(args.task, args)
        else:
            print(f"Unknown meta_model {args.meta_model}, falling back to linear")
            return create_linear_model(args.task, args)


# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
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


def train_question_models(train_df, val_df, test_df, metadata, args, best_hparams, out_dir: Path):
    """Train per-question models."""
    embedding_files = {"train": {}, "val": {}, "test": {}}
    summaries = []
    
    questions = [q.upper() for q in args.questions]
    
    # Check if test_df is empty (test_frac=0 case)
    is_test_empty = test_df is None or test_df.empty
    
    for question in questions:
        q_train = train_df[train_df["question_id"] == question].reset_index(drop=True)
        q_val = val_df[val_df["question_id"] == question].reset_index(drop=True)
        
        # Handle test DataFrame properly
        if not is_test_empty:
            q_test = test_df[test_df["question_id"] == question].reset_index(drop=True)
        else:
            q_test = pd.DataFrame()  # Empty DataFrame

        if q_train.empty:
            print(f"{question}: skipping, no training examples")
            continue

        q_dir = out_dir / "question_models" / question
        model_dir = q_dir / "model"
        train_emb = q_dir / "embeddings_train.csv"
        val_emb = q_dir / "embeddings_val.csv"
        test_emb = q_dir / "embeddings_test.csv"

        # Check if we need to train the model
        model_exists = model_dir.exists() and saved_model_exists(model_dir)
        embeddings_exist = train_emb.exists() and val_emb.exists()
        test_embeddings_exist = is_test_empty or test_emb.exists()
        
        if (model_exists and embeddings_exist and test_embeddings_exist):
            print(f"{question}: model and embeddings already exist, loading.")
        else:
            print(f"{question}: training model on {len(q_train)} examples, val on {len(q_val)}")
            q_cfg = make_question_cfg(args, question, q_dir, best_hparams)
            train_one_fold(q_train, q_val, q_cfg, metadata, q_dir)
            
            if not saved_model_exists(model_dir):
                raise FileNotFoundError(f"Expected saved model at {model_dir}")
            
            extract_embeddings(model_dir, q_train, args, train_emb, best_hparams["max_length"])
            extract_embeddings(model_dir, q_val, args, val_emb, best_hparams["max_length"])
            if not is_test_empty and not q_test.empty:
                extract_embeddings(model_dir, q_test, args, test_emb, best_hparams["max_length"])
            elif not is_test_empty:
                # Create empty test embeddings file with correct structure
                print(f"{question}: No test examples, creating empty test embeddings placeholder")
                # Create empty DataFrame with expected columns from training embeddings
                if train_emb.exists():
                    sample_emb = pd.read_csv(train_emb, nrows=1)
                    empty_test_emb = pd.DataFrame(columns=sample_emb.columns)
                    empty_test_emb.to_csv(test_emb, index=False)

        embedding_files["train"][question] = train_emb
        embedding_files["val"][question] = val_emb
        embedding_files["test"][question] = test_emb if not is_test_empty and not q_test.empty else None
        summaries.append({
            "question_id": question,
            "train_examples": len(q_train),
            "val_examples": len(q_val),
            "test_examples": len(q_test) if not is_test_empty else 0,
            "model_dir": str(model_dir)
        })
    
    pd.DataFrame(summaries).to_csv(out_dir / "question_model_summary.csv", index=False)
    return embedding_files


def build_feature_table(embedding_paths: dict[str, Path | None], questions: list[str]):
    """Build feature table from embeddings."""
    tables = []
    for q in questions:
        path = embedding_paths.get(q)
        if path is None or not Path(path).exists():
            print(f"Warning: No embeddings found for question {q}, skipping")
            continue
        try:
            emb_df = pd.read_csv(path)
            if emb_df.empty:
                print(f"Warning: Empty embeddings for question {q}, skipping")
                continue
                
            emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
            if not emb_cols:
                print(f"Warning: No embedding columns found for question {q}, skipping")
                continue
                
            grouped = emb_df.groupby("speaker_id", as_index=True).agg(
                y_true=("y_true", "first"),
                **{col: (col, "mean") for col in emb_cols},
            )
            grouped = grouped.rename(columns={col: f"{q}__{col}" for col in emb_cols})
            grouped[f"{q}__present"] = 1.0
            tables.append(grouped)
        except Exception as e:
            print(f"Error processing embeddings for question {q}: {e}")
            continue
    
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
    """Align feature columns across splits."""
    for df in [val_df, test_df]:
        if df is not None and not df.empty:
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            extra = [c for c in df.columns if "__" in c and c not in feature_cols]
            if extra:
                df.drop(columns=extra, inplace=True)
    return train_df, val_df, test_df


def question_groups(feature_cols):
    """Group features by question."""
    groups = {}
    for c in feature_cols:
        q = c.split("__", 1)[0]
        groups.setdefault(q, []).append(c)
    return groups


def permutation_question_importance(model, data_df, feature_cols, args):
    """Calculate feature importance by permuting all features from each question."""
    # This version works with a single dataset (no separate validation set)
    x_data = data_df[feature_cols].to_numpy()
    y_data = data_df["y_true"].to_numpy()
    base_metrics = score_meta_model(model, x_data, y_data, args.task)
    base_score = primary_score(base_metrics, args.task)
    groups = question_groups(feature_cols)
    rng = np.random.RandomState(args.seed)
    rows = []
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    
    for q, cols in groups.items():
        indices = [col_to_idx[c] for c in cols]
        drops = []
        for _ in range(args.permutation_repeats):
            x_perm = x_data.copy()
            shuffled = x_perm[:, indices].copy()
            rng.shuffle(shuffled)
            x_perm[:, indices] = shuffled
            m = score_meta_model(model, x_perm, y_data, args.task)
            perm_score = primary_score(m, args.task)
            drops.append(base_score - perm_score)
        rows.append({
            "question_id": q,
            "importance": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "base_score": float(base_score)
        })
    
    return pd.DataFrame(rows).sort_values("importance", ascending=False)

def shap_question_importance(model, train_df, val_df, feature_cols, args):
    """SHAP analysis that actually works with small datasets."""
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"  SHAP analysis with {len(feature_cols)} features...")
    
    n_samples = len(train_df)
    
    # CRITICAL: Use at most min(10, n_samples-1) components for stability
    n_components = min(10, n_samples - 1, len(feature_cols))
    print(f"  Using {n_components} PCA components (samples: {n_samples})")
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=args.seed)
    train_reduced = pca.fit_transform(train_df[feature_cols].to_numpy())
    val_reduced = pca.transform(val_df[feature_cols].to_numpy())
    
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Train a SIMPLE linear model on reduced features (not ensemble)
    from sklearn.linear_model import Ridge
    simple_model = Ridge(alpha=1.0, random_state=args.seed)
    simple_model.fit(train_reduced, train_df["y_true"].to_numpy())
    
    print(f"  Simple model R²: {simple_model.score(val_reduced, val_df['y_true'].to_numpy()):.3f}")
    
    # Use LinearExplainer (fast, stable, works with small data)
    print("  Creating LinearExplainer...")
    explainer = shap.LinearExplainer(simple_model, train_reduced)
    
    # Explain validation samples
    n_explain = min(20, len(val_reduced))
    val_sample = val_reduced[:n_explain]
    
    print(f"  Computing SHAP values for {n_explain} samples...")
    shap_values = explainer.shap_values(val_sample)
    
    # Get mean absolute SHAP per PCA component
    pca_importance = np.abs(shap_values).mean(axis=0)
    
    # Project back to original features
    feature_importance = np.abs(pca.components_.T @ pca_importance)
    
    # Aggregate by question
    groups = question_groups(feature_cols)
    rows = []
    feature_to_idx = {c: i for i, c in enumerate(feature_cols)}
    
    for q, cols in groups.items():
        col_indices = [feature_to_idx[c] for c in cols if c in feature_to_idx]
        if col_indices:
            importance = np.sum(feature_importance[col_indices])
            rows.append({
                "question_id": q,
                "importance": float(importance),
                "importance_std": 0.0,
                "n_features": len(col_indices)
            })
    
    importance_df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    print(f"  SHAP importance computed for {len(importance_df)} questions")
    print(f"  Top 5 questions: {importance_df.head(5)['question_id'].tolist()}")
    
    return importance_df

def permutation_question_importance_shap_hybrid(model, train_df, val_df, feature_cols, args):
    """
    Hybrid approach: Use permutation importance for feature selection,
    but also compute SHAP values for interpretation.
    """
    # First get permutation importance (as in your original code)
    perm_importance = permutation_question_importance(model, val_df, feature_cols, args)
    
    # Then compute SHAP values for the top features
    try:
        shap_importance = shap_question_importance(model, train_df, val_df, feature_cols, args)
        
        # Merge both metrics
        merged = perm_importance.merge(shap_importance, on="question_id", how="left")
        
        # Save SHAP values
        merged.to_csv(Path(args.output_dir) / "shap_question_importance.csv", index=False)
        
        return merged
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return perm_importance


def score_meta_model(model, x, y, task):
    """
    Score a meta-model.
    
    Args:
        model: The model to score (can be None if y_pred is provided directly in x?)
        x: Features or predictions
        y: True labels
        task: "classification" or "regression"
    """
    # Check if x is actually predictions (when model is None)
    if model is None:
        # Assume x contains predictions directly
        pred = x
    else:
        pred = model.predict(x)
    
    if task == "classification":
        return {
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
    else:
        rmse = np.sqrt(mean_squared_error(y, pred))
        return {
            "rmse": rmse,
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
        }


def train_meta_model(train_features, val_features, test_features, feature_cols, args, out_dir: Path):
    """Train meta-model with top-k feature selection."""
    
    # Create meta-model (single or ensemble)
    base_model = make_meta_model(args)
    base_model.fit(train_features[feature_cols].to_numpy(), train_features["y_true"].to_numpy())
    
    # Calculate feature importance
    importance_df = permutation_question_importance(base_model, val_features, feature_cols, args)
    importance_df.to_csv(out_dir / "question_embedding_importance.csv", index=False)

    questions_ranked = importance_df["question_id"].tolist()
    if not questions_ranked:
        raise ValueError("No ranked questions.")

    max_k = len(questions_ranked)
    ks = list(range(1, max_k + 1))
    if args.top_k and 0 < args.top_k < max_k:
        ks = sorted(set(ks + [args.top_k]))

    val_metrics = {}
    best_val_score = -float("inf")
    best_k = 1
    
    for k in ks:
        selected_qs = questions_ranked[:k]
        selected_cols = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs)]
        model = make_meta_model(args)
        model.fit(train_features[selected_cols].to_numpy(), train_features["y_true"].to_numpy())
        m = score_meta_model(model, val_features[selected_cols].to_numpy(), val_features["y_true"].to_numpy(), args.task)
        val_metrics[k] = m
        
        score = primary_score(m, args.task)
        if score > best_val_score:
            best_val_score = score
            best_k = k
            
    print(f"Best k on validation set: {best_k} (primary_score: {best_val_score:.4f})")

    # Train final model on train+val with best k
    trainval_features = pd.concat([train_features, val_features], ignore_index=True)
    selected_qs_final = questions_ranked[:best_k]
    selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
    final_model = make_meta_model(args)
    final_model.fit(trainval_features[selected_cols_final].to_numpy(), trainval_features["y_true"].to_numpy())

    test_metrics = score_meta_model(
        final_model,
        test_features[selected_cols_final].to_numpy(),
        test_features["y_true"].to_numpy(),
        args.task
    )
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_dir / "meta_model.joblib")
    pd.DataFrame({"question_id": selected_qs_final}).to_csv(out_dir / "selected_questions.csv", index=False)
    pd.DataFrame({"feature": selected_cols_final}).to_csv(out_dir / "selected_embedding_features.csv", index=False)
    with open(out_dir / "meta_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    preds = final_model.predict(test_features[selected_cols_final].to_numpy())
    out_df = test_features[["speaker_id", "y_true"]].copy()
    out_df["y_pred"] = preds
    
    # Get probabilities for classification
    if args.task == "classification":
        if hasattr(final_model, "predict_proba"):
            probs = final_model.predict_proba(test_features[selected_cols_final].to_numpy())
            classes = final_model.classes_
            for i, cls in enumerate(classes):
                out_df[f"prob_{cls}"] = probs[:, i]
        elif hasattr(final_model, "named_estimators_"):
            # For voting classifier, get probabilities from individual estimators
            try:
                probs = final_model.predict_proba(test_features[selected_cols_final].to_numpy())
                classes = final_model.classes_
                for i, cls in enumerate(classes):
                    out_df[f"prob_{cls}"] = probs[:, i]
            except:
                print("Warning: Could not get probabilities from ensemble")
    
    out_df.to_csv(out_dir / "meta_test_predictions.csv", index=False)

    # Save ensemble details if applicable
    if getattr(args, 'use_ensemble', False):
        ensemble_info = {
            "use_ensemble": True,
            "ensemble_models": getattr(args, 'ensemble_models', ['linear', 'random_forest', 'hist_gradient_boosting']),
            "voting_type": "soft" if args.task == "classification" else "average"
        }
        with open(out_dir / "ensemble_config.json", "w") as f:
            json.dump(ensemble_info, f, indent=2)

    val_summary = []
    for k, m in val_metrics.items():
        row = {"top_k": k, "questions": ",".join(questions_ranked[:k])}
        row["primary_score"] = primary_score(m, args.task)
        for k2, v2 in m.items():
            if isinstance(v2, (int, float)):
                row[k2] = v2
        val_summary.append(row)
    pd.DataFrame(val_summary).to_csv(out_dir / "topk_val_metrics.csv", index=False)

    return {
        "best_k": best_k,
        "best_val_primary_score": best_val_score,
        "test_metrics": test_metrics
    }

def train_meta_model_with_cv_selection(
    train_features, val_features, test_features, feature_cols, args, out_dir: Path
):
    """Train meta-model with CV-based K selection, then evaluate on test set."""
    
    # Check if CV results already exist
    cv_results_file = out_dir / "cv_k_selection_results.csv"
    selected_questions_file = out_dir / "selected_questions.csv"
    meta_model_file = out_dir / "meta_model.joblib"
    
    if cv_results_file.exists() and selected_questions_file.exists() and meta_model_file.exists() and not args.force_hpo:
        print(f"\n{'='*60}")
        print("Loading existing CV k-selection results...")
        print(f"{'='*60}")
        
        # Load existing results
        cv_results = pd.read_csv(cv_results_file)
        selected_questions_df = pd.read_csv(selected_questions_file)
        selected_qs_final = selected_questions_df["question_id"].tolist()
        
        # Load the best_k value
        best_k_row = cv_results[cv_results["is_best"] == True]
        if not best_k_row.empty:
            best_k = int(best_k_row.iloc[0]["k"])
            best_mean_score = best_k_row.iloc[0]["mean_cv_score"]
        else:
            # If no best marked, take the one with highest mean score
            best_k = int(cv_results.loc[cv_results["mean_cv_score"].idxmax(), "k"])
            best_mean_score = cv_results["mean_cv_score"].max()
        
        print(f"Loaded best K: {best_k} (CV mean score: {best_mean_score:.4f})")
        print(f"Loaded {len(selected_qs_final)} selected questions")
        
        # Build selected feature columns
        selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
        
        # Check if we need to retrain the final model
        if not meta_model_file.exists() or args.force_hpo:
            print("Retraining final model...")
            # Combine train and val
            trainval_features = pd.concat([train_features, val_features], ignore_index=True)
            
            final_model = make_meta_model(args)
            final_model.fit(
                trainval_features[selected_cols_final].to_numpy(),
                trainval_features["y_true"].to_numpy()
            )
            joblib.dump(final_model, meta_model_file)
        else:
            print("Loading existing final model...")
            final_model = joblib.load(meta_model_file)
        
        # Evaluate on held-out test set
        test_metrics = score_meta_model(
            final_model,
            test_features[selected_cols_final].to_numpy(),
            test_features["y_true"].to_numpy(),
            args.task
        )
        
        print("\nFinal test metrics (loaded from existing model):")
        print(json.dumps(test_metrics, indent=2))
        
        # Save predictions if they don't exist
        predictions_file = out_dir / "meta_test_predictions.csv"
        if not predictions_file.exists():
            preds = final_model.predict(test_features[selected_cols_final].to_numpy())
            out_df = test_features[["speaker_id", "y_true"]].copy()
            out_df["y_pred"] = preds
            
            if args.task == "classification":
                if hasattr(final_model, "predict_proba"):
                    try:
                        probs = final_model.predict_proba(test_features[selected_cols_final].to_numpy())
                        classes = final_model.classes_
                        for i, cls in enumerate(classes):
                            out_df[f"prob_{cls}"] = probs[:, i]
                    except:
                        print("Warning: Could not get probabilities")
            
            out_df.to_csv(predictions_file, index=False)
        
        return {
            "best_k": best_k,
            "best_cv_score": best_mean_score,
            "test_metrics": test_metrics,
            "loaded_from_cache": True
        }
    
    # If results don't exist, run the full CV selection
    print(f"\n{'='*60}")
    print("No existing CV results found. Running CV k-selection...")
    print(f"{'='*60}")
    
    # Combine train and val for CV-based selection
    trainval_features = pd.concat([train_features, val_features], ignore_index=True)
    
    # Create CV splits based on speakers
    speakers = trainval_features.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
    # Calculate question importance using all training+validation data
    print("\nCalculating question importance on full training+validation data...")
    base_model = make_meta_model(args)
    base_model.fit(trainval_features[feature_cols].to_numpy(), trainval_features["y_true"].to_numpy())
    
    if args.importance == "shap":
        importance_df = shap_question_importance(
            base_model, trainval_features, trainval_features, feature_cols, args
        )
    elif args.importance == "hybrid":
        importance_df = permutation_question_importance_shap_hybrid(
            base_model, trainval_features, trainval_features, feature_cols, args
        )
    else:  # permutation
        importance_df = permutation_question_importance(
            base_model, trainval_features, feature_cols, args
        )
    
    importance_df.to_csv(out_dir / "question_embedding_importance.csv", index=False)
    questions_ranked = importance_df["question_id"].tolist()
    
    # Cross-validation to find best K
    print(f"\n{'='*60}")
    print(f"Cross-validating to find best K (using {args.n_cv_folds} folds)")
    print(f"{'='*60}")
    
    ks = list(range(1, len(questions_ranked) + 1))
    if args.top_k and 0 < args.top_k < len(questions_ranked):
        ks = sorted(set(ks + [args.top_k]))
    
    # Store CV results for each K
    cv_results_by_k = {k: [] for k in ks}
    
    for fold_idx, (train_speaker_idx, val_speaker_idx) in enumerate(fold_splits):
        train_speakers = speakers.iloc[train_speaker_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_speaker_idx]["speaker_id"].values
        
        fold_train = trainval_features[trainval_features["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val = trainval_features[trainval_features["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        print(f"\nFold {fold_idx + 1}/{args.n_cv_folds}: Train={len(fold_train)}, Val={len(fold_val)}")
        
        for k in ks:
            selected_qs = questions_ranked[:k]
            selected_cols = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs)]
            
            if not selected_cols:
                cv_results_by_k[k].append(float('-inf'))
                continue
            
            # Train on fold training data
            model = make_meta_model(args)
            model.fit(fold_train[selected_cols].to_numpy(), fold_train["y_true"].to_numpy())
            
            # Evaluate on fold validation data
            metrics = score_meta_model(
                model,
                fold_val[selected_cols].to_numpy(),
                fold_val["y_true"].to_numpy(),
                args.task
            )
            
            score = primary_score(metrics, args.task)
            cv_results_by_k[k].append(score)
            print(f"  K={k}: score={score:.4f}")
    
    # Find best K based on mean CV score
    best_k = None
    best_mean_score = -float('inf')
    k_scores = {}
    
    for k, scores in cv_results_by_k.items():
        if scores and not all(s == float('-inf') for s in scores):
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            k_scores[k] = {"mean": mean_score, "std": std_score, "all_scores": scores}
            
            print(f"\nK={k}: mean CV score={mean_score:.4f} (+/- {std_score:.4f})")
            
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_k = k
    
    if best_k is None:
        print("Warning: Could not determine best K, using K=1")
        best_k = 1
    
    print(f"\n{'='*60}")
    print(f"BEST K SELECTED: {best_k} (CV mean score: {best_mean_score:.4f})")
    print(f"{'='*60}")
    
    # Save CV results
    cv_summary = []
    for k, info in k_scores.items():
        cv_summary.append({
            "k": k,
            "mean_cv_score": info["mean"],
            "std_cv_score": info["std"],
            "is_best": k == best_k
        })
    pd.DataFrame(cv_summary).to_csv(out_dir / "cv_k_selection_results.csv", index=False)
    
    # Now train final model on ALL training+validation data with best_k
    selected_qs_final = questions_ranked[:best_k]
    selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
    
    print(f"\nTraining final model on all training+validation data with K={best_k}")
    print(f"Selected questions: {selected_qs_final}")
    print(f"Number of features: {len(selected_cols_final)}")
    
    final_model = make_meta_model(args)
    final_model.fit(
        trainval_features[selected_cols_final].to_numpy(),
        trainval_features["y_true"].to_numpy()
    )
    
    # Evaluate on held-out test set
    test_metrics = score_meta_model(
        final_model,
        test_features[selected_cols_final].to_numpy(),
        test_features["y_true"].to_numpy(),
        args.task
    )
    
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    
    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_dir / "meta_model.joblib")
    pd.DataFrame({"question_id": selected_qs_final}).to_csv(out_dir / "selected_questions.csv", index=False)
    pd.DataFrame({"feature": selected_cols_final}).to_csv(out_dir / "selected_embedding_features.csv", index=False)
    with open(out_dir / "meta_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save predictions
    preds = final_model.predict(test_features[selected_cols_final].to_numpy())
    out_df = test_features[["speaker_id", "y_true"]].copy()
    out_df["y_pred"] = preds
    
    if args.task == "classification":
        if hasattr(final_model, "predict_proba"):
            try:
                probs = final_model.predict_proba(test_features[selected_cols_final].to_numpy())
                classes = final_model.classes_
                for i, cls in enumerate(classes):
                    out_df[f"prob_{cls}"] = probs[:, i]
            except:
                print("Warning: Could not get probabilities")
    
    out_df.to_csv(out_dir / "meta_test_predictions.csv", index=False)
    
    # Save ensemble info if used
    if getattr(args, 'use_ensemble', False):
        ensemble_info = {
            "use_ensemble": True,
            "ensemble_models": getattr(args, 'ensemble_models', []),
            "voting_type": "soft" if args.task == "classification" else "average",
            "best_k": best_k,
            "best_cv_score": best_mean_score,
            "selected_questions": selected_qs_final
        }
        with open(out_dir / "ensemble_config.json", "w") as f:
            json.dump(ensemble_info, f, indent=2)
    
    return {
        "best_k": best_k,
        "best_cv_score": best_mean_score,
        "test_metrics": test_metrics,
        "cv_results_by_k": k_scores,
        "loaded_from_cache": False
    }

# ----------------------------------------------------------------------
#  Parser Setup
# ----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Train per‑question models with ensemble meta-model")
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--questions", nargs="+", default=[f"Q{i}" for i in range(1, 15)])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hyperparameters
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    
    # HPO settings
    parser.add_argument("--hpo-backend", choices=["grid", "random", "optuna"], default="optuna")
    parser.add_argument("--hpo-n-trials", type=int, default=20)  # Changed to 20
    parser.add_argument("--hpo-timeout", type=int, default=None)
    parser.add_argument("--hpo-folds", type=int, default=3)
    parser.add_argument("--force-hpo", action="store_true")
    
    # Meta-model settings - SINGLE MODEL
    parser.add_argument("--meta-model", 
                        choices=["linear", "random_forest", "svm", "hist_gradient_boosting", "gradient_boosting", "knn",
                                "ridge", "lasso", "elasticnet"],
                        default="linear",
                        help="Single meta-model to use (ignored if --use-ensemble is set)")
    
    # Ensemble settings
    parser.add_argument("--use-ensemble", action="store_true",
                        help="Use ensemble of multiple meta-models with voting/averaging")
    parser.add_argument("--ensemble-models", nargs="+",
                        choices=["linear", "random_forest", "svm", "hist_gradient_boosting", "gradient_boosting", "knn",
                                "ridge", "lasso", "elasticnet"],
                        default=["linear", "random_forest", "hist_gradient_boosting"],
                        help="Models to include in ensemble")
    parser.add_argument("--ensemble-weights", nargs="+", type=float, default=None,
                        help="Custom weights for ensemble members (default: equal weights)")
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--logreg-C", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--svm-kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--importance", choices=["shap", "permutation", "hybrid"], default="permutation")
    parser.add_argument("--svm-C", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-epsilon", type=float, default=0.1)
    parser.add_argument("--xgb-lr", type=float, default=0.1)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--elasticnet-alpha", type=float, default=1.0)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    
    # Other settings
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--delimiter", default=";")
    
    return parser

def train_meta_model_cv(
    train_features, val_features, test_features, feature_cols, args, out_dir: Path
):
    """Train meta-model with cross-validation only (test_percentage=0)."""
    
    # Combine train and val for CV
    all_trainval = pd.concat([train_features, val_features], ignore_index=True)
    
    # Create CV splits based on speakers
    speakers = all_trainval.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
    fold_results = []
    all_fold_predictions = []
    fold_importance_dfs = []
    best_k_per_fold = []
    
    for fold_idx, (train_speaker_idx, val_speaker_idx) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/{args.n_cv_folds}")
        print(f"{'='*50}")
        
        train_speakers = speakers.iloc[train_speaker_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_speaker_idx]["speaker_id"].values
        
        fold_train = all_trainval[all_trainval["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val = all_trainval[all_trainval["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        print(f"Train size: {len(fold_train)}, Val size: {len(fold_val)}")
        
        # Train base model for importance calculation
        base_model = make_meta_model(args)
        base_model.fit(fold_train[feature_cols].to_numpy(), fold_train["y_true"].to_numpy())

        # Calculate question importance for this fold
        if args.importance == "shap":
            importance_df = shap_question_importance(
                base_model, fold_train, fold_val, feature_cols, args
            )

        elif args.importance == "hybrid":
            importance_df = permutation_question_importance_shap_hybrid(
                base_model, fold_train, fold_val, feature_cols, args
            )
        else: 
            importance_df = permutation_question_importance(
                base_model, fold_val, feature_cols, args
            )
        importance_df["fold"] = fold_idx
        fold_importance_dfs.append(importance_df)
        
        # Get ranked questions for this fold
        questions_ranked = importance_df["question_id"].tolist()
        if not questions_ranked:
            questions_ranked = [c.split("__", 1)[0] for c in feature_cols]
        
        max_k = len(questions_ranked)
        ks = list(range(1, max_k + 1))
        if args.top_k and 0 < args.top_k < max_k:
            ks = sorted(set(ks + [args.top_k]))
        
        # Find best k for this fold
        best_val_score = -float("inf")
        best_k = 1
        fold_val_metrics = {}
        
        for k in ks:
            selected_qs = questions_ranked[:k]
            selected_cols = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs)]
            
            if not selected_cols:
                continue
                
            model = make_meta_model(args)
            model.fit(fold_train[selected_cols].to_numpy(), fold_train["y_true"].to_numpy())
            metrics = score_meta_model(
                model, 
                fold_val[selected_cols].to_numpy(), 
                fold_val["y_true"].to_numpy(), 
                args.task
            )
            fold_val_metrics[k] = metrics
            
            score = primary_score(metrics, args.task)
            if score > best_val_score:
                best_val_score = score
                best_k = k
        
        best_k_per_fold.append({"fold": fold_idx, "best_k": best_k, "best_score": best_val_score})
        print(f"Best k for fold {fold_idx}: {best_k} (score: {best_val_score:.4f})")
        
        # Train final model for this fold with best k
        selected_qs_final = questions_ranked[:best_k]
        selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
        
        final_model = make_meta_model(args)
        final_model.fit(
            fold_train[selected_cols_final].to_numpy(), 
            fold_train["y_true"].to_numpy()
        )
        
        # Evaluate on validation set
        val_metrics = score_meta_model(
            final_model,
            fold_val[selected_cols_final].to_numpy(),
            fold_val["y_true"].to_numpy(),
            args.task
        )
        
        # Store predictions
        val_preds = final_model.predict(fold_val[selected_cols_final].to_numpy())
        fold_predictions = fold_val[["speaker_id", "y_true"]].copy()
        fold_predictions["y_pred"] = val_preds
        fold_predictions["fold"] = fold_idx
        
        if args.task == "classification" and hasattr(final_model, "predict_proba"):
            try:
                probs = final_model.predict_proba(fold_val[selected_cols_final].to_numpy())
                classes = final_model.classes_
                for i, cls in enumerate(classes):
                    fold_predictions[f"prob_{cls}"] = probs[:, i]
            except:
                pass
        
        all_fold_predictions.append(fold_predictions)
        
        fold_results.append({
            "fold": fold_idx,
            "best_k": best_k,
            "val_metrics": val_metrics,
            "selected_questions": selected_qs_final,
            "n_selected_features": len(selected_cols_final)
        })
        
        # Save individual fold model
        fold_model_dir = out_dir / f"fold_{fold_idx}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, fold_model_dir / "meta_model.joblib")
        pd.DataFrame({"question_id": selected_qs_final}).to_csv(
            fold_model_dir / "selected_questions.csv", index=False
        )
    
    # Aggregate results across folds
    print("\n" + "="*60)
    print("AGGREGATED RESULTS ACROSS FOLDS")
    print("="*60)
    
    # Combine all predictions
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(out_dir / "cv_all_predictions.csv", index=False)
    
    # Calculate aggregate metrics using the stored predictions
    # Don't pass a model, pass the predictions directly
    aggregate_metrics = score_meta_model(
        None,  # No model needed
        all_predictions["y_pred"].values,  # Pass predictions as x
        all_predictions["y_true"].values,  # Pass true labels as y
        args.task
    )
    
    print("\nAggregate metrics across all folds:")
    print(json.dumps(aggregate_metrics, indent=2))
    
    # Per-fold statistics
    fold_summaries = []
    for res in fold_results:
        score = primary_score(res["val_metrics"], args.task)
        fold_summaries.append({
            "fold": res["fold"],
            "best_k": res["best_k"],
            "primary_score": score,
            "n_selected_questions": len(res["selected_questions"])
        })
    
    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(out_dir / "fold_summary.csv", index=False)
    
    print("\nPer-fold results:")
    print(fold_summary_df)
    print(f"\nMean best_k: {fold_summary_df['best_k'].mean():.1f}")
    print(f"Mean primary score: {fold_summary_df['primary_score'].mean():.4f} (+/- {fold_summary_df['primary_score'].std():.4f})")
    
    # Aggregate question importance across folds
    all_importance = pd.concat(fold_importance_dfs, ignore_index=True)
    
    # Calculate mean importance per question across folds
    question_importance_agg = all_importance.groupby("question_id").agg({
        "importance": ["mean", "std", "count"],
        "importance_std": "mean"
    }).round(4)
    question_importance_agg.columns = ["mean_importance", "std_importance", "n_folds", "mean_importance_std"]
    question_importance_agg = question_importance_agg.sort_values("mean_importance", ascending=False)
    question_importance_agg.to_csv(out_dir / "aggregated_question_importance.csv")
    
    print("\nTop 10 most important questions across folds:")
    print(question_importance_agg.head(10))
    
    # Determine final top K based on average best_k across folds
    avg_best_k = int(np.round(fold_summary_df['best_k'].mean()))
    print(f"\nAverage best K across folds: {avg_best_k}")
    
    # Get top K questions based on mean importance
    top_questions = question_importance_agg.head(avg_best_k).index.tolist()
    top_features = [c for c in feature_cols if c.split("__", 1)[0] in top_questions]
    
    print(f"Selected {len(top_questions)} top questions: {top_questions}")
    
    # Train final ensemble model on all data using top K questions
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON ALL DATA WITH TOP K QUESTIONS")
    print("="*60)
    
    final_model = make_meta_model(args)
    final_model.fit(
        all_trainval[top_features].to_numpy(),
        all_trainval["y_true"].to_numpy()
    )
    
    # Save final model
    joblib.dump(final_model, out_dir / "final_cv_model.joblib")
    pd.DataFrame({"question_id": top_questions}).to_csv(
        out_dir / "final_selected_questions.csv", index=False
    )
    pd.DataFrame({"feature": top_features}).to_csv(
        out_dir / "final_selected_features.csv", index=False
    )
    
    # Save aggregate results
    cv_results = {
        "cv_folds": args.n_cv_folds,
        "aggregate_metrics": aggregate_metrics,
        "per_fold_metrics": fold_summaries,
        "avg_best_k": avg_best_k,
        "selected_questions": top_questions,
        "mean_primary_score": float(fold_summary_df['primary_score'].mean()),
        "std_primary_score": float(fold_summary_df['primary_score'].std())
    }
    
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    # Create summary report
    with open(out_dir / "cv_summary_report.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CROSS-VALIDATION SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of folds: {args.n_cv_folds}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Total samples: {len(all_trainval)}\n\n")
        
        f.write("Per-fold Results:\n")
        f.write("-"*40 + "\n")
        for res in fold_summaries:
            f.write(f"Fold {res['fold']}: best_k={res['best_k']}, primary_score={res['primary_score']:.4f}\n")
        
        f.write(f"\nAverage Results:\n")
        f.write(f"  Mean best_k: {avg_best_k}\n")
        f.write(f"  Mean primary score: {fold_summary_df['primary_score'].mean():.4f} (+/- {fold_summary_df['primary_score'].std():.4f})\n\n")
        
        f.write("Aggregate Metrics Across All Folds:\n")
        f.write("-"*40 + "\n")
        for k, v in aggregate_metrics.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")
            elif k == "classification_report" and isinstance(v, dict):
                f.write(f"  {k}:\n")
                for class_label, metrics in v.items():
                    if isinstance(metrics, dict):
                        f.write(f"    {class_label}: {metrics}\n")
        
        f.write("\nTop 10 Most Important Questions:\n")
        f.write("-"*40 + "\n")
        for idx, (q, row) in enumerate(question_importance_agg.head(10).iterrows(), 1):
            f.write(f"  {idx}. {q}: importance={row['mean_importance']:.4f} (+/- {row['std_importance']:.4f})\n")
        
        f.write(f"\nFinal Selected Questions (K={avg_best_k}):\n")
        f.write("-"*40 + "\n")
        for i, q in enumerate(top_questions, 1):
            f.write(f"  {i}. {q}\n")
    
    print(f"\nResults saved to {out_dir}")
    print(f"  - cv_all_predictions.csv: All fold predictions")
    print(f"  - cv_results.json: Aggregate results")
    print(f"  - cv_summary_report.txt: Detailed summary report")
    print(f"  - final_cv_model.joblib: Final model trained on all data")
    print(f"  - aggregated_question_importance.csv: Question importance across folds")
    
    return {
        "cv_folds": args.n_cv_folds,
        "aggregate_metrics": aggregate_metrics,
        "per_fold_metrics": fold_summaries,
        "avg_best_k": avg_best_k,
        "selected_questions": top_questions,
        "question_importance": question_importance_agg.to_dict()
    }

def main_with_cv(args, train_features, val_features, test_features, feature_cols, out_dir):
    """Run CV-only training when test_frac == 0"""
    
    # Check if test_features is empty (which it will be when test_frac=0)
    if test_features is not None and not test_features.empty:
        print(f"Warning: test_features is not empty but test_frac=0. This shouldn't happen.")
    
    # Train meta-model with CV (since test_frac=0)
    results = train_meta_model_cv(
        train_features, val_features, test_features, feature_cols, args, out_dir
    )
    
    print("\n===== Final CV Results =====")
    print(json.dumps(results, indent=2))
    
    # Cleanup
    cleanup_temp_dirs(out_dir)
    
    # Print ensemble info if used
    if args.use_ensemble:
        print("\n" + "=" * 50)
        print("ENSEMBLE SUMMARY")
        print("=" * 50)
        print(f"Models in ensemble: {args.ensemble_models}")
        print(f"Voting type: {'soft (probability averaging)' if args.task == 'classification' else 'average'}")
        print(f"Final CV model saved to: {out_dir / 'final_cv_model.joblib'}") 


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    if args.test_frac == 0:
        # For CV-only mode, check for CV summary report
        cv_summary_file = out_dir / "cv_summary_report.txt"
        cv_results_file = out_dir / "cv_results.json"
        
        if cv_summary_file.exists() and cv_results_file.exists() and not args.force_hpo:
            print("\n" + "="*60)
            print("FINAL CV SUMMARY ALREADY EXISTS")
            print("="*60)
            print(f"Found existing CV results at:")
            print(f"  - {cv_summary_file}")
            print(f"  - {cv_results_file}")
            print(f"\nSkipping all processing.")
            print(f"To re-run, either:")
            print(f"  1. Delete {out_dir} directory, or")
            print(f"  2. Use --force-hpo flag")
            print("="*60 + "\n")
            return
    else:
        # For standard mode with test set, check for test metrics file
        test_metrics_file = out_dir / "meta_test_metrics.json"
        selected_questions_file = out_dir / "selected_questions.csv"
        
        if test_metrics_file.exists() and selected_questions_file.exists() and not args.force_hpo:
            print("\n" + "="*60)
            print("FINAL MODEL ALREADY EXISTS")
            print("="*60)
            print(f"Found existing final model results at:")
            print(f"  - {test_metrics_file}")
            print(f"  - {selected_questions_file}")
            print(f"\nSkipping all processing.")
            print(f"To re-run, either:")
            print(f"  1. Delete {out_dir} directory, or")
            print(f"  2. Use --force-hpo flag")
            print("="*60 + "\n")
            
            # Optionally load and display existing results
            if test_metrics_file.exists():
                import json
                with open(test_metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
                print("Existing test metrics:")
                print(json.dumps(existing_metrics, indent=2))
            return
    
    
    cleanup_old_splits(splits_dir)
    
    # Save config
    (out_dir / "question_ensemble_config.json").write_text(json.dumps(vars(args), indent=2))
    
    # Load full dataset
    questions = [q.upper() for q in args.questions]
    df, metadata = load_examples(
        args.asr_file, args.demo_file, args.target_column, args.task,
        text_mode="question", min_text_chars=args.min_text_chars,
        filter_questions=questions, delimiter=args.delimiter
    )
    
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # Manage splits
    split_mgr = SplitManager(
        splits_dir, args.task,
        args.train_frac, args.val_frac, args.test_frac,
        args.seed, args.n_cv_folds
    )
    final_train, final_val, final_test = split_mgr.get_final_splits(df)
    print(f"Final splits: train={len(final_train)}, val={len(final_val)}, test={len(final_test)}")
    
    # Define best_hparams_path BEFORE using it
    best_hparams_path = out_dir / "best_hyperparams_all_questions.json"
    
    # Hyperparameter search on ALL questions
    if best_hparams_path.exists() and not args.force_hpo:
        best_hparams = json.loads(best_hparams_path.read_text())
        print(f"Loaded best hyperparameters from {best_hparams_path}")
        print(f"Best parameters: {best_hparams}")
    else:
        try:
            best_hparams = hyperparameter_search_optuna_all_questions(
                final_train, split_mgr, args, metadata, final_test
            )
            best_hparams_path.write_text(json.dumps(best_hparams, indent=2))
            print(f"Saved best hyperparameters to {best_hparams_path}")
        except Exception as e:
            print(f"HPO failed: {e}")
            print("Using default hyperparameters")
            best_hparams = {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "max_length": args.max_length,
            }
            print(f"Default parameters: {best_hparams}")
    
    # Update args with best hyperparameters
    args.learning_rate = best_hparams["learning_rate"]
    args.batch_size = best_hparams["batch_size"]
    args.epochs = best_hparams["epochs"]
    args.weight_decay = best_hparams.get("weight_decay", args.weight_decay)
    args.warmup_ratio = best_hparams.get("warmup_ratio", args.warmup_ratio)
    args.max_length = best_hparams.get("max_length", args.max_length)
    
    # Train per-question models
    embedding_files = train_question_models(
        final_train, final_val, final_test, metadata, args, best_hparams, out_dir
    )
    
    # Build feature tables
    available_qs = list(embedding_files["train"].keys())
    train_features, feature_cols = build_feature_table(embedding_files["train"], available_qs)
    val_features, _ = build_feature_table(embedding_files["val"], available_qs)
    
    # Handle test features carefully
    test_features = None
    if args.test_frac > 0:
        test_features, _ = build_feature_table(embedding_files["test"], available_qs)
    else:
        print("test_frac=0: No test features will be created")
        # Create empty test_features with same structure as train_features but no rows
        if train_features is not None and not train_features.empty:
            test_features = pd.DataFrame(columns=train_features.columns)
        else:
            test_features = pd.DataFrame()
    
    # Save raw feature tables
    train_features.to_csv(out_dir / "meta_train_features.csv", index=False)
    val_features.to_csv(out_dir / "meta_val_features.csv", index=False)
    
    # Align feature columns
    if test_features is not None and not test_features.empty:
        train_features, val_features, test_features = align_feature_tables(
            train_features, val_features, test_features, feature_cols
        )
    else:
        # Just align train and val
        train_features, val_features, _ = align_feature_tables(
            train_features, val_features, pd.DataFrame(), feature_cols
        )
    
    # ============================================================
    # CHOOSE TRAINING PATH BASED ON test_frac
    # ============================================================
    if args.test_frac == 0:
        # Cross-validation only (no held-out test set)
        main_with_cv(args, train_features, val_features, test_features, feature_cols, out_dir)
    else: 
        # Standard training with held-out test set
        test_features.to_csv(out_dir / "meta_test_features.csv", index=False)
        
        # Train meta-model WITH CV-based K selection (UPDATED)
        results = train_meta_model_with_cv_selection(
            train_features, val_features, test_features, feature_cols, args, out_dir
        )
        
        # ============================================================
        # THESE PRINT STATEMENTS ARE IN THE ELSE PART
        # ============================================================
        print("\n===== Final Results =====")
        print(json.dumps(results, indent=2))
        
        # Cleanup
        cleanup_temp_dirs(out_dir)
        
        # Print ensemble info if used
        if args.use_ensemble:
            print("\n" + "=" * 50)
            print("ENSEMBLE SUMMARY")
            print("=" * 50)
            print(f"Models in ensemble: {args.ensemble_models}")
            print(f"Voting type: {'soft (probability averaging)' if args.task == 'classification' else 'average'}")
            print(f"Ensemble saved to: {out_dir / 'meta_model.joblib'}")


if __name__ == "__main__":
    main()

'''
1. Multiple Meta-Model Options:
linear - Logistic Regression (classification) / Ridge (regression)

random_forest - Random Forest

svm - Support Vector Machine

xgboost - XGBoost

gradient_boosting - Gradient Boosting

knn - K-Nearest Neighbors

2. Ensemble Support:
bash
# Use ensemble with default models (linear, random_forest, xgboost)
python train.py --use-ensemble --task classification ...

# Specify custom ensemble members
python train.py --use-ensemble --ensemble-models linear svm xgboost --task classification ...

# Use custom weights for ensemble members
python train.py --use-ensemble --ensemble-models linear random_forest xgboost --ensemble-weights 1.0 2.0 1.5
3. How Ensemble Works:
For Classification (Soft Voting):

python
# Averages probabilities from all models
final_prob = (prob_linear + prob_rf + prob_xgb) / 3
prediction = argmax(final_prob)
For Regression (Averaging):

python
# Averages predictions from all models
final_prediction = (pred_linear + pred_rf + pred_xgb) / 3
4. Usage Examples:
bash
# Single model (original behavior)
python train.py --meta-model xgboost --task classification

# Ensemble of 3 models (default)
python train.py --use-ensemble --task classification --hpo-n-trials 20

# Ensemble with custom weights (give more weight to XGBoost)
python train.py --use-ensemble \
  --ensemble-models linear random_forest xgboost \
  --ensemble-weights 1.0 1.0 2.0 \
  --task classification

# Regression ensemble
python train.py --use-ensemble \
  --ensemble-models ridge random_forest xgboost \
  --task regression
5. Benefits of Ensemble:
✅ More robust - Less likely to overfit

✅ Better generalization - Combines strengths of different algorithms

✅ Handles different data patterns - Each model captures different relationships

✅ Default 20 trials - Faster hyperparameter search


Pipeline Overview (with Nested Cross‑Validation)
1. Speaker‑Level Data Splitting
The full dataset is split into train+val (90%) and test (10%) using speaker‑level stratification (StratifiedShuffleSplit for classification, ShuffleSplit for regression).

The test set is locked away and never used until final evaluation.

From the 90% train+val portion, a further speaker‑level split creates a validation set (10% of the original) and the actual training set (80% of original). This validation set is used for early stopping and feature selection.

2. Nested Cross‑Validation for Hyperparameter Optimisation (HPO)
The pipeline uses a nested CV strategy to avoid overfitting when tuning the per‑question Transformer models:

Inner CV (Optuna):
For each trial, a K‑fold cross‑validation (default 3 folds) is performed on the train+val data (90% of the original).

Each fold respects speaker‑level grouping.

The model is trained on fold_train and evaluated on fold_val for all questions simultaneously.

The primary metric (macro F1 for classification, negative RMSE for regression) is averaged across folds and questions.

This average guides the Optuna sampler (TPE) to propose new hyperparameter sets.

The search yields optimal hyperparameters (learning rate, batch size, epochs, etc.) without ever touching the final test set.

No outer CV on the test set:
The test set is only used once at the very end for final evaluation, ensuring unbiased generalisation estimates.

3. Per‑Question Model Training (with Best Hyperparameters)
Using the optimal hyperparameters from Step 2, a separate Transformer model is trained for each question on the training portion (80% of original) and validated on the validation set (10% of original).

Early stopping prevents overfitting.

For each trained model, embeddings are extracted from the last hidden layer (mean‑pooled over tokens) for all splits (train, val, test).

4. Meta‑Model Feature Construction
For each speaker, embeddings from each question are averaged, producing a feature vector:
[Q1_emb1, Q1_emb2, …, Q2_emb1, …] plus a present flag per question.

The resulting tables for train, val, and test are merged and aligned.

5. Meta‑Model Training with Embedded Feature Selection
A base meta‑model (e.g., Logistic Regression, Random Forest, or an ensemble) is trained on the train features to compute permutation importance per question.

Questions are ranked by importance, and a top‑k selection is performed using the validation set – the k that maximises the primary score on the validation set is chosen.

Finally, the meta‑model is retrained on the combined train+val features (90% of original data) using only the selected top‑k question embeddings.

6. Final Evaluation on the Held‑Out Test Set
The final meta‑model is evaluated once on the test features (10% of original data).

Metrics are saved, and predictions (with probabilities for classification) are written to disk.

Why Nested Cross‑Validation?
Level   Purpose Data Used
Inner CV    Tune per‑question Transformer hyperparameters   Train+val (90%) – cross‑validation splits
Validation set  Select top‑k questions and early stop   Separate validation set (10% of original)
Test set    Final, unbiased performance estimate    Held‑out 10% – used only once
This design prevents any information from the test set leaking into model selection or hyperparameter tuning, giving you a trustworthy estimate of real‑world performance.

'''