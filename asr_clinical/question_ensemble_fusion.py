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
    roc_auc_score
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
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA

# Cache directory to avoid redownloading
import os
os.environ['HF_HOME'] = '/home/bahman/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/bahman/.cache/huggingface/transformers'


# =======================================================================
#  UTILITY FUNCTIONS
# =======================================================================

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
    else:
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


def download_with_retry(model_name: str, max_retries: int = 5):
    """Download model with retry logic for rate limiting."""
    from transformers import AutoConfig
    import time
    
    for retry in range(max_retries):
        try:
            config = AutoConfig.from_pretrained(model_name)
            return config
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (retry + 1) * 10
                print(f"Rate limited! Waiting {wait_time} seconds before retry {retry+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(f"Failed to download {model_name} after {max_retries} retries")


# =======================================================================
#  SPLIT MANAGEMENT
# =======================================================================

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
        
        if self.test_frac == 0:
            print("test_frac=0: No test set will be created.")
            rel_val_frac = self.val_frac / (self.train_frac + self.val_frac)
            train_idx, val_idx = self._speaker_split(df, rel_val_frac, self.seed)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = pd.DataFrame()
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            if len(df) > 0:
                empty_test = pd.DataFrame(columns=df.columns)
                empty_test.to_csv(test_path, index=False)
            
            print("Final splits saved (train/val only).")
            return train_df, val_df, test_df
        
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
        if test_size <= 0:
            return np.arange(len(df)), np.array([], dtype=int)
        
        if test_size >= 1:
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


# =======================================================================
#  PRIMARY SCORE FUNCTION
# =======================================================================

def primary_score(metrics: dict, task: str) -> float:
    if task == "classification":
        return metrics.get("macro_f1", 0.0)
    else:
        return -metrics.get("rmse", float('inf'))


def score_meta_model(model, x, y, task):
    """Score a meta-model. Computes AUC if model supports predict_proba."""
    if model is None:
        pred = x
        proba = None
    else:
        pred = model.predict(x)
        proba = model.predict_proba(x) if hasattr(model, "predict_proba") else None

    if task == "classification":
        metrics = {
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
        if proba is not None:
            try:
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y, proba[:, 1])
                else:
                    metrics["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class='ovr', average='macro')
            except Exception:
                pass
        return metrics
    else:
        rmse = np.sqrt(mean_squared_error(y, pred))
        return {
            "rmse": rmse,
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
        }


# =======================================================================
#  HYPERPARAMETER SEARCH
# =======================================================================

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
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "epochs": trial.suggest_int("epochs", 1, 3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "max_length": trial.suggest_categorical("max_length", [128, 256]),
        "focal_gamma": trial.suggest_float("focal_gamma", 0.5, 5.0, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.3),
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.1, 5.0, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
    }
    
    print(f"\nTrial {trial.number}: testing {params}")
    
    all_question_scores = []
    
    for fold_idx, (fold_train, fold_val) in enumerate(folds):
        fold_question_scores = []
        
        for question in all_questions:
            q_fold_train = fold_train[fold_train["question_id"] == question].reset_index(drop=True)
            q_fold_val = fold_val[fold_val["question_id"] == question].reset_index(drop=True)
            
            if len(q_fold_train) < 10 or len(q_fold_val) < 3:
                print(f"  Skipping {question}: train={len(q_fold_train)}, val={len(q_fold_val)} (insufficient data)")
                continue
            
            temp_out = Path(args.output_dir) / "temp_hpo_optuna" / f"trial{trial.number}_fold{fold_idx}_{question}"
            
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
                        patience=1,
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
                        wait_time = (retry + 1) * 5
                        print(f"  Rate limited! Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    continue
                finally:
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


def _train_and_evaluate_fast(train_df, val_df, cfg: TrainConfig, metadata: dict) -> dict | None:
    """Fast training and evaluation for hyperparameter search with caching."""
    from transformers import AutoModelForSequenceClassification
    from .model import load_tokenizer
    from .train import choose_device, train_one_fold, saved_model_exists
    
    cache_key = f"{hash(frozenset(train_df['utterance_id']))}_{cfg.learning_rate}_{cfg.batch_size}_{cfg.epochs}"
    cache_path = Path(cfg.output_dir) / f"cache_{cache_key}.pkl"
    
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
        batch_size = min(cfg.eval_batch_size, len(texts))

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
        
        joblib.dump(result, cache_path)
        return result
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None


# =======================================================================
#  META-MODEL CREATORS
# =======================================================================

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
    else:
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
    else:
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
                probability=True,
                class_weight="balanced",
                random_state=args.seed
            )),
        ])
    else:
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
    else:
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
    else:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=getattr(args, 'knn_neighbors', 5),
                weights='distance'
            )),
        ])


def create_ensemble_model(task, args):
    """Create an ensemble of multiple meta-models with voting/averaging."""
    
    ensemble_models = getattr(args, 'ensemble_models', ['linear', 'random_forest', 'xgboost'])
    
    regression_only_models = ['ridge', 'lasso', 'elasticnet']
    
    if task == "classification":
        invalid_models = [m for m in ensemble_models if m in regression_only_models]
        if invalid_models:
            print(f"Warning: Removing regression-only models from classification ensemble: {invalid_models}")
            ensemble_models = [m for m in ensemble_models if m not in regression_only_models]
    
    if not ensemble_models:
        print("Error: No valid models for ensemble. Falling back to linear model.")
        return create_linear_model(task, args)
    
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
    failed_models = []
    
    print(f"\nCreating ensemble for {task} task with models: {ensemble_models}")
    
    for model_name in ensemble_models:
        if model_name not in model_creators:
            print(f"  ✗ Unknown model '{model_name}', skipping")
            failed_models.append(model_name)
            continue
        
        try:
            model = model_creators[model_name](task, args)
            
            if task == "classification":
                from sklearn.base import ClassifierMixin
                if not isinstance(model, ClassifierMixin):
                    if hasattr(model, 'named_steps'):
                        last_step = list(model.named_steps.values())[-1]
                        if not isinstance(last_step, ClassifierMixin):
                            raise ValueError(f"{model_name} is not a classifier (got {type(last_step).__name__})")
                    else:
                        raise ValueError(f"{model_name} is not a classifier (got {type(model).__name__})")
            else:
                from sklearn.base import RegressorMixin
                if not isinstance(model, RegressorMixin):
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
    
    successful_models = [m for m in ensemble_models if m not in failed_models]
    
    if not estimators:
        print("\nNo valid ensemble models. Falling back to linear model.")
        return create_linear_model(task, args)
    
    if task == "classification":
        voting_type = getattr(args, 'ensemble_voting', 'soft')
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            weights=getattr(args, 'ensemble_weights', None),
            n_jobs=-1
        )
        
        print(f"\n✓ Created CLASSIFICATION ensemble with {len(estimators)} models")
        print(f"  - Voting type: {voting_type}")
        print(f"  - Models: {successful_models}")
        
        if getattr(args, 'ensemble_weights', None):
            print(f"  - Weights: {args.ensemble_weights}")
        
    else:
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=getattr(args, 'ensemble_weights', None),
            n_jobs=-1
        )
        
        print(f"\n✓ Created REGRESSION ensemble with {len(estimators)} models")
        print(f"  - Models: {successful_models}")
        
        if getattr(args, 'ensemble_weights', None):
            print(f"  - Weights: {args.ensemble_weights}")
    
    args.ensemble_models_used = successful_models
    
    return ensemble


def make_meta_model(args):
    """Create meta-model (single or ensemble based on configuration)."""
    
    if getattr(args, 'use_ensemble', False):
        print("\n" + "=" * 50)
        print("CREATING ENSEMBLE META-MODEL")
        print("=" * 50)
        return create_ensemble_model(args.task, args)
    else:
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


# =======================================================================
#  EMBEDDING EXTRACTION HELPERS
# =======================================================================

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


# =======================================================================
#  PER-QUESTION MODEL TRAINING
# =======================================================================

def train_question_models(train_df, val_df, test_df, metadata, args, best_hparams, out_dir: Path):
    """Train per-question models."""
    embedding_files = {"train": {}, "val": {}, "test": {}}
    summaries = []
    
    questions = [q.upper() for q in args.questions]
    is_test_empty = test_df is None or test_df.empty
    
    validation_scores = {}
    
    for question in questions:
        q_train = train_df[train_df["question_id"] == question].reset_index(drop=True)
        q_val = val_df[val_df["question_id"] == question].reset_index(drop=True)
        
        if not is_test_empty:
            q_test = test_df[test_df["question_id"] == question].reset_index(drop=True)
        else:
            q_test = pd.DataFrame()

        if q_train.empty:
            print(f"{question}: skipping, no training examples")
            continue

        q_dir = out_dir / "question_models" / question
        model_dir = q_dir / "model"
        train_emb = q_dir / "embeddings_train.csv"
        val_emb = q_dir / "embeddings_val.csv"
        test_emb = q_dir / "embeddings_test.csv"

        model_exists = model_dir.exists() and saved_model_exists(model_dir)
        embeddings_exist = train_emb.exists() and val_emb.exists()
        test_embeddings_exist = is_test_empty or test_emb.exists()
        
        if (model_exists and embeddings_exist and test_embeddings_exist):
            print(f"{question}: model and embeddings already exist, loading.")
            
            summary_path = out_dir / "question_model_summary.csv"
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                if not summary_df.empty and 'val_score' in summary_df.columns:
                    row = summary_df[summary_df['question_id'] == question]
                    if not row.empty and not pd.isna(row['val_score'].iloc[0]):
                        validation_scores[question] = row['val_score'].iloc[0]
        else:
            print(f"{question}: training model on {len(q_train)} examples, val on {len(q_val)}")
            q_cfg = make_question_cfg(args, question, q_dir, best_hparams)
            train_one_fold(q_train, q_val, q_cfg, metadata, q_dir)
            
            if not saved_model_exists(model_dir):
                raise FileNotFoundError(f"Expected saved model at {model_dir}")
            
            extract_embeddings(model_dir, q_train, args, train_emb, best_hparams["max_length"])
            extract_embeddings(model_dir, q_val, args, val_emb, best_hparams["max_length"])
            
            try:
                device = choose_device()
                tokenizer = load_tokenizer(str(model_dir))
                model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
                model.eval()
                
                texts = q_val["text"].tolist()
                labels = q_val["label"].values
                preds = []
                batch_size = min(best_hparams.get("batch_size", getattr(args, 'batch_size', 8)), len(texts))
                
                with torch.no_grad():
                    for start in range(0, len(texts), batch_size):
                        batch_texts = texts[start:start+batch_size]
                        enc = tokenizer(
                            batch_texts,
                            truncation=True,
                            padding=True,
                            max_length=best_hparams.get("max_length", getattr(args, 'max_length', 256)),
                            return_tensors="pt",
                        )
                        enc = {k: v.to(device) for k, v in enc.items()}
                        outputs = model(**enc)
                        logits = outputs.logits.cpu().numpy()
                        
                        if args.task == "classification":
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
                
                if len(preds) > 0 and len(labels) > 0:
                    if args.task == "classification":
                        val_score = f1_score(labels, preds, average="macro", zero_division=0)
                    else:
                        val_score = np.sqrt(mean_squared_error(labels, preds))
                    
                    validation_scores[question] = val_score
                    print(f"  {question} validation score: {val_score:.4f}")
                else:
                    print(f"  {question}: No predictions or labels to evaluate")
                    validation_scores[question] = None
                
            except Exception as e:
                print(f"  Could not calculate validation score for {question}: {e}")
                validation_scores[question] = None
            
            if not is_test_empty and not q_test.empty:
                extract_embeddings(model_dir, q_test, args, test_emb, best_hparams["max_length"])
            elif not is_test_empty:
                print(f"{question}: No test examples, creating empty test embeddings placeholder")
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
            "val_score": validation_scores.get(question, None),
            "model_dir": str(model_dir)
        })
    
    if validation_scores:
        valid_scores = {q: s for q, s in validation_scores.items() if s is not None}
        
        if valid_scores:
            scores_df = pd.DataFrame([
                {"question_id": q, "validation_score": s}
                for q, s in valid_scores.items()
            ])
            
            if not scores_df.empty:
                scores_df = scores_df.sort_values("validation_score", ascending=False)
                scores_df.to_csv(out_dir / "per_question_model_validation_scores.csv", index=False)
                print(f"\n✓ Per-question model validation scores saved to: {out_dir / 'per_question_model_validation_scores.csv'}")
    
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


# =======================================================================
#  IMPORTANCE CALCULATION FUNCTIONS
# =======================================================================

def permutation_question_importance(model, data_df, feature_cols, args):
    """Calculate feature importance by permuting all features from each question."""
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
    """SHAP analysis that works with small datasets."""
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"  SHAP analysis with {len(feature_cols)} features...")
    
    n_samples = len(train_df)
    n_components = min(10, n_samples - 1, len(feature_cols))
    print(f"  Using {n_components} PCA components (samples: {n_samples})")
    
    pca = PCA(n_components=n_components, random_state=args.seed)
    train_reduced = pca.fit_transform(train_df[feature_cols].to_numpy())
    val_reduced = pca.transform(val_df[feature_cols].to_numpy())
    
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    simple_model = Ridge(alpha=1.0, random_state=args.seed)
    simple_model.fit(train_reduced, train_df["y_true"].to_numpy())
    
    print(f"  Simple model R²: {simple_model.score(val_reduced, val_df['y_true'].to_numpy()):.3f}")
    
    print("  Creating LinearExplainer...")
    explainer = shap.LinearExplainer(simple_model, train_reduced)
    
    n_explain = min(20, len(val_reduced))
    val_sample = val_reduced[:n_explain]
    
    print(f"  Computing SHAP values for {n_explain} samples...")
    shap_values = explainer.shap_values(val_sample)
    
    pca_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = np.abs(pca.components_.T @ pca_importance)
    
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
    """Hybrid approach: Use permutation importance for feature selection, plus SHAP."""
    perm_importance = permutation_question_importance(model, val_df, feature_cols, args)
    
    try:
        shap_importance = shap_question_importance(model, train_df, val_df, feature_cols, args)
        merged = perm_importance.merge(shap_importance, on="question_id", how="left")
        merged.to_csv(Path(args.output_dir) / "shap_question_importance.csv", index=False)
        return merged
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return perm_importance


def save_per_question_validation_scores(
    train_features, val_features, feature_cols, 
    questions_ranked, args, out_dir: Path, 
    prefix: str = ""
):
    """Calculate and save validation score for each question individually."""
    print("\n" + "="*60)
    print("CALCULATING PER-QUESTION VALIDATION SCORES")
    print("="*60)
    
    per_question_scores = []
    
    for idx, q in enumerate(questions_ranked):
        q_cols = [c for c in feature_cols if c.split("__", 1)[0] == q]
        
        if not q_cols:
            print(f"  Skipping {q}: no features found")
            continue
        
        print(f"  Evaluating {q} (rank {idx+1}/{len(questions_ranked)}) with {len(q_cols)} features...")
        
        model = make_meta_model(args)
        
        try:
            model.fit(
                train_features[q_cols].to_numpy(), 
                train_features["y_true"].to_numpy()
            )
            
            metrics = score_meta_model(
                model,
                val_features[q_cols].to_numpy(),
                val_features["y_true"].to_numpy(),
                args.task
            )
            
            score = primary_score(metrics, args.task)
            
            additional_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ['primary_score']:
                    additional_metrics[k] = v
            
            per_question_scores.append({
                "question_id": q,
                "rank": idx + 1,
                "validation_score": score,
                "n_features": len(q_cols),
                **additional_metrics
            })
            
        except Exception as e:
            print(f"    Error evaluating {q}: {e}")
            continue
    
    df = pd.DataFrame(per_question_scores)
    if not df.empty:
        df = df.sort_values("validation_score", ascending=False)
        df['rank_by_score'] = range(1, len(df) + 1)
        
        filename = f"{prefix}_per_question_validation_scores.csv" if prefix else "per_question_validation_scores.csv"
        df.to_csv(out_dir / filename, index=False)
        
        print(f"\n✓ Saved per-question validation scores to: {out_dir / filename}")
        print(f"\nTop 10 questions by validation score:")
        print(df.head(10)[["question_id", "validation_score", "rank", "rank_by_score"]].to_string(index=False))
        
        summary = {
            "mean_score": df['validation_score'].mean(),
            "std_score": df['validation_score'].std(),
            "min_score": df['validation_score'].min(),
            "max_score": df['validation_score'].max(),
            "n_questions": len(df)
        }
        with open(out_dir / f"{prefix}_per_question_score_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return df
    else:
        print("  No valid question scores could be calculated")
        return None


# =======================================================================
#  AUDIO FEATURE LOADING AND FUSION UTILITIES
# =======================================================================

def load_audio_features(csv_path: str, speaker_col: str = "speaker_id",
                        exclude_cols: list = None) -> pd.DataFrame:
    """Load audio features from CSV and prepare them for fusion."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Audio features CSV not found: {csv_path}")
    
    audio_df = pd.read_csv(csv_path)
    
    if speaker_col not in audio_df.columns:
        raise ValueError(f"Speaker column '{speaker_col}' not found in audio CSV. Available: {audio_df.columns.tolist()}")
    
    exclude = exclude_cols or [speaker_col, 'session_id', 'utterance_id', 'question_id', 'label', 'y_true']
    numeric_cols = audio_df.select_dtypes(include=[np.number]).columns.tolist()
    audio_feature_cols = [c for c in numeric_cols if c not in exclude]
    
    if not audio_feature_cols:
        raise ValueError("No numeric feature columns found in audio CSV. Check exclude_cols.")
    
    keep_cols = [speaker_col] + audio_feature_cols
    audio_df = audio_df[keep_cols].copy()
    
    if audio_df.duplicated(subset=[speaker_col]).any():
        audio_df = audio_df.groupby(speaker_col).mean().reset_index()
    
    if speaker_col != 'speaker_id':
        audio_df.rename(columns={speaker_col: 'speaker_id'}, inplace=True)
    
    print(f"Loaded audio features: {len(audio_df)} speakers, {len(audio_feature_cols)} features")
    print(f"  Audio feature columns: {audio_feature_cols[:5]}...")
    return audio_df


def merge_audio_features(feature_df: pd.DataFrame, audio_df: pd.DataFrame,
                         how: str = 'left') -> pd.DataFrame:
    """Merge speaker-level feature table with audio features."""
    merged = feature_df.merge(audio_df, on='speaker_id', how=how)
    if how == 'left':
        audio_cols = [c for c in audio_df.columns if c != 'speaker_id']
        for col in audio_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
    return merged


def get_audio_feature_cols(audio_df: pd.DataFrame) -> list:
    """Return list of audio feature column names (excludes speaker_id)."""
    return [c for c in audio_df.columns if c != 'speaker_id']


# =======================================================================
#  TRAIN_META_MODEL FUNCTIONS
# =======================================================================

def train_meta_model_with_cv_selection(
    train_features, val_features, test_features, feature_cols, args, out_dir: Path
):
    """Train meta-model with CV-based K selection, then evaluate on test set."""
    
    cv_results_file = out_dir / "cv_k_selection_results.csv"
    selected_questions_file = out_dir / "selected_questions.csv"
    meta_model_file = out_dir / "meta_model.joblib"
    
    if cv_results_file.exists() and selected_questions_file.exists() and meta_model_file.exists() and not args.force_hpo:
        print(f"\n{'='*60}")
        print("Loading existing CV k-selection results...")
        print(f"{'='*60}")
        
        cv_results = pd.read_csv(cv_results_file)
        selected_questions_df = pd.read_csv(selected_questions_file)
        selected_qs_final = selected_questions_df["question_id"].tolist()
        
        best_k_row = cv_results[cv_results["is_best"] == True]
        if not best_k_row.empty:
            best_k = int(best_k_row.iloc[0]["k"])
            best_mean_score = best_k_row.iloc[0]["mean_cv_score"]
        else:
            best_k = int(cv_results.loc[cv_results["mean_cv_score"].idxmax(), "k"])
            best_mean_score = cv_results["mean_cv_score"].max()
        
        print(f"Loaded best K: {best_k} (CV mean score: {best_mean_score:.4f})")
        print(f"Loaded {len(selected_qs_final)} selected questions")
        
        selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
        
        if not meta_model_file.exists() or args.force_hpo:
            print("Retraining final model...")
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
        
        test_metrics = score_meta_model(
            final_model,
            test_features[selected_cols_final].to_numpy(),
            test_features["y_true"].to_numpy(),
            args.task
        )
        
        print("\nFinal test metrics (loaded from existing model):")
        print(json.dumps(test_metrics, indent=2))
        
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
    
    print(f"\n{'='*60}")
    print("No existing CV results found. Running CV k-selection...")
    print(f"{'='*60}")
    
    trainval_features = pd.concat([train_features, val_features], ignore_index=True)
    
    speakers = trainval_features.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
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
    else:
        importance_df = permutation_question_importance(
            base_model, trainval_features, feature_cols, args
        )
    
    importance_df.to_csv(out_dir / "question_embedding_importance.csv", index=False)
    questions_ranked = importance_df["question_id"].tolist()

    per_question_scores = save_per_question_validation_scores(
        train_features, val_features, feature_cols, 
        questions_ranked, args, out_dir, 
        prefix="initial"
    )
    
    print(f"\n{'='*60}")
    print(f"Cross-validating to find best K (using {args.n_cv_folds} folds)")
    print(f"{'='*60}")
    
    ks = list(range(1, len(questions_ranked) + 1))
    if args.top_k and 0 < args.top_k < len(questions_ranked):
        ks = sorted(set(ks + [args.top_k]))
    
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
            
            model = make_meta_model(args)
            model.fit(fold_train[selected_cols].to_numpy(), fold_train["y_true"].to_numpy())
            
            metrics = score_meta_model(
                model,
                fold_val[selected_cols].to_numpy(),
                fold_val["y_true"].to_numpy(),
                args.task
            )
            
            score = primary_score(metrics, args.task)
            cv_results_by_k[k].append(score)
            print(f"  K={k}: score={score:.4f}")
    
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
    
    cv_summary = []
    for k, info in k_scores.items():
        cv_summary.append({
            "k": k,
            "mean_cv_score": info["mean"],
            "std_cv_score": info["std"],
            "is_best": k == best_k
        })
    pd.DataFrame(cv_summary).to_csv(out_dir / "cv_k_selection_results.csv", index=False)
    
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


def train_meta_model_cv(
    train_features, val_features, test_features, feature_cols, args, out_dir: Path
):
    """Train meta-model with cross-validation only (test_percentage=0)."""
    
    all_trainval = pd.concat([train_features, val_features], ignore_index=True)
    
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
    all_per_question_scores = []
    
    for fold_idx, (train_speaker_idx, val_speaker_idx) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/{args.n_cv_folds}")
        print(f"{'='*50}")
        
        train_speakers = speakers.iloc[train_speaker_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_speaker_idx]["speaker_id"].values
        
        fold_train = all_trainval[all_trainval["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val = all_trainval[all_trainval["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        print(f"Train size: {len(fold_train)}, Val size: {len(fold_val)}")
        
        base_model = make_meta_model(args)
        base_model.fit(fold_train[feature_cols].to_numpy(), fold_train["y_true"].to_numpy())

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
        
        questions_ranked = importance_df["question_id"].tolist()
        if not questions_ranked:
            questions_ranked = [c.split("__", 1)[0] for c in feature_cols]

        fold_scores = save_per_question_validation_scores(
            fold_train, fold_val, feature_cols, 
            questions_ranked, args, out_dir, 
            prefix=f"fold{fold_idx}"
        )
        if fold_scores is not None:
            fold_scores['fold'] = fold_idx
            all_per_question_scores.append(fold_scores)
        
        max_k = len(questions_ranked)
        ks = list(range(1, max_k + 1))
        if args.top_k and 0 < args.top_k < max_k:
            ks = sorted(set(ks + [args.top_k]))
        
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
        
        print(f"Best k for fold {fold_idx}: {best_k} (score: {best_val_score:.4f})")
        
        selected_qs_final = questions_ranked[:best_k]
        selected_cols_final = [c for c in feature_cols if c.split("__", 1)[0] in set(selected_qs_final)]
        
        final_model = make_meta_model(args)
        final_model.fit(
            fold_train[selected_cols_final].to_numpy(), 
            fold_train["y_true"].to_numpy()
        )
        
        val_metrics = score_meta_model(
            final_model,
            fold_val[selected_cols_final].to_numpy(),
            fold_val["y_true"].to_numpy(),
            args.task
        )
        
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
        
        fold_model_dir = out_dir / f"fold_{fold_idx}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, fold_model_dir / "meta_model.joblib")
        pd.DataFrame({"question_id": selected_qs_final}).to_csv(
            fold_model_dir / "selected_questions.csv", index=False
        )
    
    if all_per_question_scores:
        combined_scores = pd.concat(all_per_question_scores, ignore_index=True)
        
        agg_scores = combined_scores.groupby('question_id').agg({
            'validation_score': ['mean', 'std', 'count'],
            'rank': 'mean',
            'rank_by_score': 'mean'
        }).round(4)
        agg_scores.columns = ['mean_score', 'std_score', 'n_folds', 'mean_rank', 'mean_rank_by_score']
        agg_scores = agg_scores.sort_values('mean_score', ascending=False)
        agg_scores.to_csv(out_dir / "aggregated_per_question_validation_scores.csv")
        
        print("\n" + "="*60)
        print("AGGREGATED PER-QUESTION VALIDATION SCORES ACROSS FOLDS")
        print("="*60)
        print("\nTop 10 questions by mean validation score:")
        print(agg_scores.head(10)[['mean_score', 'std_score', 'n_folds']].to_string())

    print("\n" + "="*60)
    print("AGGREGATED RESULTS ACROSS FOLDS")
    print("="*60)
    
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_predictions.to_csv(out_dir / "cv_all_predictions.csv", index=False)
    
    aggregate_metrics = score_meta_model(
        None,
        all_predictions["y_pred"].values,
        all_predictions["y_true"].values,
        args.task
    )
    
    print("\nAggregate metrics across all folds:")
    print(json.dumps(aggregate_metrics, indent=2))
    
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
    
    all_importance = pd.concat(fold_importance_dfs, ignore_index=True)
    
    question_importance_agg = all_importance.groupby("question_id").agg({
        "importance": ["mean", "std", "count"],
        "importance_std": "mean"
    }).round(4)
    question_importance_agg.columns = ["mean_importance", "std_importance", "n_folds", "mean_importance_std"]
    question_importance_agg = question_importance_agg.sort_values("mean_importance", ascending=False)
    question_importance_agg.to_csv(out_dir / "aggregated_question_importance.csv")
    
    print("\nTop 10 most important questions across folds:")
    print(question_importance_agg.head(10))
    
    avg_best_k = int(np.round(fold_summary_df['best_k'].mean()))
    print(f"\nAverage best K across folds: {avg_best_k}")
    
    top_questions = question_importance_agg.head(avg_best_k).index.tolist()
    top_features = [c for c in feature_cols if c.split("__", 1)[0] in top_questions]
    
    print(f"Selected {len(top_questions)} top questions: {top_questions}")
    
    selected_questions_df = pd.DataFrame({"question_id": top_questions})
    selected_questions_df.to_csv(out_dir / "selected_questions.csv", index=False)
    print(f"\n✓ Saved selected_questions.csv with {len(top_questions)} questions")
    
    cv_k_results = []
    for fold_idx in range(args.n_cv_folds):
        fold_best_k = fold_summary_df[fold_summary_df['fold'] == fold_idx]['best_k'].values
        if len(fold_best_k) > 0:
            fold_score = fold_summary_df[fold_summary_df['fold'] == fold_idx]['primary_score'].values[0]
            cv_k_results.append({
                "k": int(fold_best_k[0]),
                "mean_cv_score": fold_score,
                "std_cv_score": 0.0,
                "is_best": fold_best_k[0] == avg_best_k,
                "fold": fold_idx
            })
    
    cv_k_results.append({
        "k": avg_best_k,
        "mean_cv_score": fold_summary_df['primary_score'].mean(),
        "std_cv_score": fold_summary_df['primary_score'].std(),
        "is_best": True,
        "fold": -1
    })
    
    cv_k_results_df = pd.DataFrame(cv_k_results)
    cv_k_results_df.to_csv(out_dir / "cv_k_selection_results.csv", index=False)
    print(f"✓ Saved cv_k_selection_results.csv with {len(cv_k_results)} entries")
    
    cv_metrics_for_json = {
        "cv_aggregate_metrics": aggregate_metrics,
        "mean_cv_score": float(fold_summary_df['primary_score'].mean()),
        "std_cv_score": float(fold_summary_df['primary_score'].std()),
        "avg_best_k": avg_best_k,
        "per_fold_scores": fold_summary_df['primary_score'].tolist(),
        "per_fold_best_k": fold_summary_df['best_k'].tolist(),
        "note": "These are cross-validation metrics (no held-out test set because test_frac=0)",
        "n_folds": args.n_cv_folds,
        "total_samples": len(all_trainval)
    }
    
    if args.task == "classification":
        cv_metrics_for_json["macro_f1"] = aggregate_metrics.get("macro_f1", 0.0)
        cv_metrics_for_json["weighted_f1"] = aggregate_metrics.get("weighted_f1", 0.0)
        cv_metrics_for_json["balanced_accuracy"] = aggregate_metrics.get("balanced_accuracy", 0.0)
    else:
        cv_metrics_for_json["rmse"] = aggregate_metrics.get("rmse", float('inf'))
        cv_metrics_for_json["mae"] = aggregate_metrics.get("mae", float('inf'))
        cv_metrics_for_json["r2"] = aggregate_metrics.get("r2", 0.0)
    
    with open(out_dir / "meta_test_metrics.json", "w") as f:
        json.dump(cv_metrics_for_json, f, indent=2)
    print(f"✓ Saved meta_test_metrics.json with CV aggregate metrics")
    
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON ALL DATA WITH TOP K QUESTIONS")
    print("="*60)
    
    final_model = make_meta_model(args)
    final_model.fit(
        all_trainval[top_features].to_numpy(),
        all_trainval["y_true"].to_numpy()
    )
    
    joblib.dump(final_model, out_dir / "final_cv_model.joblib")
    pd.DataFrame({"question_id": top_questions}).to_csv(
        out_dir / "final_selected_questions.csv", index=False
    )
    pd.DataFrame({"feature": top_features}).to_csv(
        out_dir / "final_selected_features.csv", index=False
    )
    
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
    
    with open(out_dir / "cv_summary_report.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CROSS-VALIDATION SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of folds: {args.n_cv_folds}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Total samples: {len(all_trainval)}\n")
        f.write(f"Average best K: {avg_best_k}\n\n")
        
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
    
    print(f"\n✓ Results saved to {out_dir}")
    print(f"  - selected_questions.csv: Top {len(top_questions)} questions selected")
    print(f"  - cv_k_selection_results.csv: K selection results from CV")
    print(f"  - meta_test_metrics.json: CV aggregate metrics")
    print(f"  - final_cv_model.joblib: Final model trained on all data")
    print(f"  - cv_all_predictions.csv: All fold predictions")
    print(f"  - cv_results.json: Aggregate results")
    print(f"  - cv_summary_report.txt: Detailed summary report")
    print(f"  - aggregated_question_importance.csv: Question importance across folds")
    
    if args.use_ensemble:
        print("\n" + "=" * 50)
        print("ENSEMBLE SUMMARY")
        print("=" * 50)
        print(f"Models in ensemble: {args.ensemble_models}")
        print(f"Voting type: {'soft (probability averaging)' if args.task == 'classification' else 'average'}")
        print(f"Final CV model saved to: {out_dir / 'final_cv_model.joblib'}")
    
    return {
        "cv_folds": args.n_cv_folds,
        "aggregate_metrics": aggregate_metrics,
        "per_fold_metrics": fold_summaries,
        "avg_best_k": avg_best_k,
        "selected_questions": top_questions,
        "question_importance": question_importance_agg.to_dict(),
        "cv_k_selection_results": cv_k_results,
        "cv_metrics": cv_metrics_for_json
    }


# =======================================================================
#  ALL FUSION METHODS - COMPLETE IMPLEMENTATIONS
# =======================================================================

def run_text_only_baseline(train_features, val_features, test_features,
                           feature_cols_text, args, out_dir: Path,
                           text_meta_model=None, text_selected_features=None):
    """TEXT ONLY baseline - reuses existing text model if available."""
    text_dir = out_dir / "text_only"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    if text_meta_model is not None and text_selected_features is not None:
        print("  ✓ Using existing text model from main pipeline")
        joblib.dump(text_meta_model, text_dir / "meta_model.joblib")
        pd.DataFrame({"feature": text_selected_features}).to_csv(
            text_dir / "selected_embedding_features.csv", index=False
        )
        
        if test_features is not None and not test_features.empty:
            X_test = test_features[text_selected_features].to_numpy()
            y_test = test_features["y_true"].to_numpy()
            test_metrics = score_meta_model(text_meta_model, X_test, y_test, args.task)
        else:
            test_metrics = {"macro_f1": 0.0, "note": "CV mode - use main pipeline results"}
        
        result = {
            "test_metrics": test_metrics,
            "used_existing_model": True,
            "n_selected_features": len(text_selected_features)
        }
        
        with open(text_dir / "fusion_metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    else:
        if args.test_frac == 0:
            result = train_meta_model_cv(
                train_features, val_features, test_features, 
                feature_cols_text, args, text_dir
            )
        else:
            result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features, 
                feature_cols_text, args, text_dir
            )
        
        with open(text_dir / "fusion_metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        return result


def run_audio_only_baseline(train_features, val_features, test_features,
                            audio_feature_cols, args, out_dir: Path):
    """
    AUDIO ONLY baseline - trains on audio features only.
    Returns None if training fails (caller will stop execution).
    """
    audio_dir = out_dir / "audio_only"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Training audio-only model with {len(audio_feature_cols)} audio features")
    
    # Check if audio features are usable
    if len(audio_feature_cols) == 0:
        print("  ❌ No audio features available!")
        return None
    
    # Check if training data has enough samples
    if len(train_features) < 10:
        print(f"  ❌ Not enough training samples: {len(train_features)} (need at least 10)")
        return None
    
    # Check if there are enough unique speakers
    unique_speakers = train_features['speaker_id'].nunique()
    if unique_speakers < args.n_cv_folds:
        print(f"  ❌ Not enough unique speakers: {unique_speakers} (need at least {args.n_cv_folds} for CV)")
        return None
    
    try:
        if args.test_frac == 0:
            result = train_meta_model_cv(
                train_features, val_features, test_features, 
                audio_feature_cols, args, audio_dir
            )
        else:
            result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features, 
                audio_feature_cols, args, audio_dir
            )
        
        # Check if result is valid
        if result is None:
            print("  ❌ Audio-only training produced None result")
            return None
        
        if isinstance(result, dict) and not result:
            print("  ❌ Audio-only training produced empty result")
            return None
        
        # Check for successful metrics
        if args.task == "classification":
            if "macro_f1" not in result and "test_metrics" not in result:
                print(f"  ❌ Audio-only result missing 'macro_f1' or 'test_metrics'")
                print(f"     Result keys: {result.keys()}")
                return None
        else:
            if "rmse" not in result and "test_metrics" not in result:
                print(f"  ❌ Audio-only result missing 'rmse' or 'test_metrics'")
                print(f"     Result keys: {result.keys()}")
                return None
        
        # Save result
        with open(audio_dir / "fusion_metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Extract score for logging
        if args.task == "classification":
            if "test_metrics" in result:
                score = result["test_metrics"].get("macro_f1", 0.0)
            else:
                score = result.get("macro_f1", 0.0)
        else:
            if "test_metrics" in result:
                score = result["test_metrics"].get("rmse", float('inf'))
            else:
                score = result.get("rmse", float('inf'))
        
        print(f"  ✅ Audio-only completed with {len(audio_feature_cols)} features, score: {score:.4f}")
        return result
        
    except Exception as e:
        print(f"  ❌ Audio-only training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_early_fusion(train_features, val_features, test_features,
                     feature_cols_text, audio_feature_cols, args, out_dir: Path):
    """EARLY FUSION - concatenates text and audio features."""
    early_dir = out_dir / "early_fusion"
    early_dir.mkdir(parents=True, exist_ok=True)
    
    early_feature_cols = feature_cols_text + audio_feature_cols
    print(f"  Early fusion features: {len(early_feature_cols)} total")
    print(f"    Text: {len(feature_cols_text)}, Audio: {len(audio_feature_cols)}")
    
    if args.test_frac == 0:
        result = train_meta_model_cv(
            train_features, val_features, test_features, 
            early_feature_cols, args, early_dir
        )
    else:
        result = train_meta_model_with_cv_selection(
            train_features, val_features, test_features, 
            early_feature_cols, args, early_dir
        )
    
    with open(early_dir / "fusion_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result

# =======================================================================
#  FIXED: LATE FUSION - Properly uses audio features
# =======================================================================

def run_late_fusion(train_features, val_features, test_features,
                    train_with_audio, val_with_audio, test_with_audio,
                    feature_cols_text, audio_feature_cols, args, out_dir: Path,
                    text_meta_model=None, text_selected_features=None):
    """
    LATE FUSION - averages predictions from separate text and audio models.
    Uses audio features properly.
    """
    late_dir = out_dir / "late_fusion"
    late_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_frac == 0:
        # CV-based late fusion - PASS audio features
        metrics, preds, y_true = run_late_fusion_cv(
            train_features, val_features,
            train_with_audio, val_with_audio,
            feature_cols_text, audio_feature_cols, args, late_dir
        )
        with open(late_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        # Held-out test mode
        # Text model (NO audio)
        if text_meta_model is not None and text_selected_features is not None:
            print("  Using existing text model for late fusion...")
            text_model = text_meta_model
            text_selected = text_selected_features
        else:
            print("  Training text model for late fusion...")
            text_result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features,
                feature_cols_text, args, late_dir / "text_model"
            )
            text_model = joblib.load(late_dir / "text_model" / "meta_model.joblib")
            text_selected = pd.read_csv(late_dir / "text_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Audio model (ONLY audio) - USING train_with_audio, val_with_audio, test_with_audio
        print("  Training audio model for late fusion...")
        audio_result = train_meta_model_with_cv_selection(
            train_with_audio, val_with_audio, test_with_audio,
            audio_feature_cols, args, late_dir / "audio_model"
        )
        audio_model = joblib.load(late_dir / "audio_model" / "meta_model.joblib")
        audio_selected = pd.read_csv(late_dir / "audio_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Get predictions on test set
        X_text_test = test_features[text_selected].to_numpy()
        X_audio_test = test_with_audio[audio_selected].to_numpy()  # Audio features
        y_true = test_features["y_true"].to_numpy()
        
        if args.task == "classification":
            if hasattr(text_model, "predict_proba") and hasattr(audio_model, "predict_proba"):
                prob_text = text_model.predict_proba(X_text_test)
                prob_audio = audio_model.predict_proba(X_audio_test)
                prob_avg = (prob_text + prob_audio) / 2.0
                preds = np.argmax(prob_avg, axis=1)
            else:
                pred_text = text_model.predict(X_text_test)
                pred_audio = audio_model.predict(X_audio_test)
                preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0,
                                            arr=np.array([pred_text, pred_audio]))
        else:
            pred_text = text_model.predict(X_text_test)
            pred_audio = audio_model.predict(X_audio_test)
            preds = (pred_text + pred_audio) / 2.0
        
        metrics = score_meta_model(None, preds, y_true, args.task)
        
        with open(late_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = preds
        pred_df.to_csv(late_dir / "predictions.csv", index=False)
        
        return metrics


def run_late_fusion_cv(train_features, val_features,
                       train_with_audio, val_with_audio,
                       feature_cols_text, audio_feature_cols, args, out_dir):
    """Late fusion using CV with audio features."""
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
    all_preds = []
    all_true = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_speakers = speakers.iloc[train_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_idx]["speaker_id"].values
        
        # Text data
        fold_train = all_text_data[all_text_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val = all_text_data[all_text_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Audio data
        fold_train_audio = all_audio_data[all_audio_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val_audio = all_audio_data[all_audio_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Train text model
        text_result = train_meta_model_cv(
            fold_train, fold_val, None, feature_cols_text, args, 
            out_dir / f"late_text_fold{fold_idx}"
        )
        text_model = joblib.load(out_dir / f"late_text_fold{fold_idx}" / "final_cv_model.joblib")
        text_selected = pd.read_csv(out_dir / f"late_text_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        # Train audio model - USING audio data
        audio_result = train_meta_model_cv(
            fold_train_audio, fold_val_audio, None, audio_feature_cols, args, 
            out_dir / f"late_audio_fold{fold_idx}"
        )
        audio_model = joblib.load(out_dir / f"late_audio_fold{fold_idx}" / "final_cv_model.joblib")
        audio_selected = pd.read_csv(out_dir / f"late_audio_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        X_text_val = fold_val[text_selected].to_numpy()
        X_audio_val = fold_val_audio[audio_selected].to_numpy()
        y_val = fold_val["y_true"].to_numpy()
        
        if args.task == "classification":
            if hasattr(text_model, "predict_proba") and hasattr(audio_model, "predict_proba"):
                prob_text = text_model.predict_proba(X_text_val)
                prob_audio = audio_model.predict_proba(X_audio_val)
                prob_avg = (prob_text + prob_audio) / 2.0
                preds = np.argmax(prob_avg, axis=1)
            else:
                pred_text = text_model.predict(X_text_val)
                pred_audio = audio_model.predict(X_audio_val)
                preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0,
                                            arr=np.array([pred_text, pred_audio]))
        else:
            pred_text = text_model.predict(X_text_val)
            pred_audio = audio_model.predict(X_audio_val)
            preds = (pred_text + pred_audio) / 2.0
        
        fold_metrics.append(score_meta_model(None, preds, y_val, args.task))
        all_preds.extend(preds)
        all_true.extend(y_val)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "late_fusion_fold_metrics.csv", index=False)
    
    pred_df = all_text_data[["speaker_id", "y_true"]].copy()
    pred_df["y_pred"] = all_preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    return aggregate_metrics, all_preds, all_true


# =======================================================================
#  FIXED: MODEL-BASED FUSION - Properly uses audio features
# =======================================================================

def run_model_based_fusion(train_features, val_features, test_features,
                           train_with_audio, val_with_audio, test_with_audio,
                           feature_cols_text, audio_feature_cols, args, out_dir: Path,
                           text_meta_model=None, text_selected_features=None):
    """
    MODEL-BASED FUSION (Stacking) - meta-model on predictions from text and audio models.
    Uses audio features properly.
    """
    mbf_dir = out_dir / "model_based_fusion"
    mbf_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_frac == 0:
        metrics, preds, y_true = run_model_based_fusion_cv(
            train_features, val_features,
            train_with_audio, val_with_audio,
            feature_cols_text, audio_feature_cols, args, mbf_dir
        )
        with open(mbf_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        # Text model
        if text_meta_model is not None and text_selected_features is not None:
            print("  Using existing text model for stacking...")
            text_model = text_meta_model
            text_selected = text_selected_features
        else:
            print("  Training text model for stacking...")
            text_result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features,
                feature_cols_text, args, mbf_dir / "text_model"
            )
            text_model = joblib.load(mbf_dir / "text_model" / "meta_model.joblib")
            text_selected = pd.read_csv(mbf_dir / "text_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Audio model - USING train_with_audio, val_with_audio, test_with_audio
        print("  Training audio model for stacking...")
        audio_result = train_meta_model_with_cv_selection(
            train_with_audio, val_with_audio, test_with_audio,
            audio_feature_cols, args, mbf_dir / "audio_model"
        )
        audio_model = joblib.load(mbf_dir / "audio_model" / "meta_model.joblib")
        audio_selected = pd.read_csv(mbf_dir / "audio_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Get predictions - text uses text features, audio uses audio features
        train_text_X = train_features[text_selected].to_numpy()
        train_audio_X = train_with_audio[audio_selected].to_numpy()
        val_text_X = val_features[text_selected].to_numpy()
        val_audio_X = val_with_audio[audio_selected].to_numpy()
        test_text_X = test_features[text_selected].to_numpy()
        test_audio_X = test_with_audio[audio_selected].to_numpy()
        
        if args.task == "classification":
            classes = np.unique(train_features["y_true"])
            n_classes = len(classes)
            
            def get_probs(model, X):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    return np.eye(n_classes)[preds.astype(int)]
            
            train_probs_text = get_probs(text_model, train_text_X)
            train_probs_audio = get_probs(audio_model, train_audio_X)
            val_probs_text = get_probs(text_model, val_text_X)
            val_probs_audio = get_probs(audio_model, val_audio_X)
            test_probs_text = get_probs(text_model, test_text_X)
            test_probs_audio = get_probs(audio_model, test_audio_X)
            
            X_train_meta = np.concatenate([train_probs_text, train_probs_audio], axis=1)
            X_val_meta = np.concatenate([val_probs_text, val_probs_audio], axis=1)
            X_test_meta = np.concatenate([test_probs_text, test_probs_audio], axis=1)
        else:
            train_preds_text = text_model.predict(train_text_X)
            train_preds_audio = audio_model.predict(train_audio_X)
            val_preds_text = text_model.predict(val_text_X)
            val_preds_audio = audio_model.predict(val_audio_X)
            test_preds_text = text_model.predict(test_text_X)
            test_preds_audio = audio_model.predict(test_audio_X)
            
            X_train_meta = np.column_stack([train_preds_text, train_preds_audio])
            X_val_meta = np.column_stack([val_preds_text, val_preds_audio])
            X_test_meta = np.column_stack([test_preds_text, test_preds_audio])
        
        y_train = train_features["y_true"].to_numpy()
        y_val = val_features["y_true"].to_numpy()
        y_test = test_features["y_true"].to_numpy()
        
        # Train meta-model on train + val
        X_meta_all = np.concatenate([X_train_meta, X_val_meta], axis=0)
        y_meta_all = np.concatenate([y_train, y_val], axis=0)
        
        if args.task == "classification":
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=args.seed))
            ])
        else:
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        
        meta_model.fit(X_meta_all, y_meta_all)
        y_pred = meta_model.predict(X_test_meta)
        metrics = score_meta_model(None, y_pred, y_test, args.task)
        
        with open(mbf_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = y_pred
        if args.task == "classification" and hasattr(meta_model, "predict_proba"):
            probs = meta_model.predict_proba(X_test_meta)
            classes = meta_model.classes_
            for i, cls in enumerate(classes):
                pred_df[f"prob_{cls}"] = probs[:, i]
        pred_df.to_csv(mbf_dir / "predictions.csv", index=False)
        
        return metrics


def run_model_based_fusion_cv(train_features, val_features,
                              train_with_audio, val_with_audio,
                              feature_cols_text, audio_feature_cols, args, out_dir):
    """Model-based fusion using CV with audio features."""
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv_outer = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(cv_outer.split(speakers, speakers["label"]))
    else:
        cv_outer = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(cv_outer.split(speakers))
    
    all_preds = []
    all_true = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
        train_speakers = speakers.iloc[train_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_idx]["speaker_id"].values
        
        # Text data
        outer_train = all_text_data[all_text_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        outer_val = all_text_data[all_text_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Audio data
        outer_train_audio = all_audio_data[all_audio_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        outer_val_audio = all_audio_data[all_audio_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Inner CV for OOF predictions
        inner_folds = min(3, len(outer_train) // 10) if len(outer_train) > 10 else 2
        inner_folds = max(2, inner_folds)
        n_train = len(outer_train)
        
        if args.task == "classification":
            classes = np.unique(outer_train["y_true"])
            n_classes = len(classes)
            oof_text_probs = np.zeros((n_train, n_classes))
            oof_audio_probs = np.zeros((n_train, n_classes))
        else:
            oof_text_preds = np.zeros(n_train)
            oof_audio_preds = np.zeros(n_train)
        
        inner_speakers = outer_train.groupby("speaker_id")["y_true"].first().reset_index()
        inner_speakers.columns = ["speaker_id", "label"]
        
        if args.task == "classification":
            cv_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=args.seed+fold_idx)
            inner_splits = list(cv_inner.split(inner_speakers, inner_speakers["label"]))
        else:
            cv_inner = KFold(n_splits=inner_folds, shuffle=True, random_state=args.seed+fold_idx)
            inner_splits = list(cv_inner.split(inner_speakers))
        
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_train_speakers = inner_speakers.iloc[inner_train_idx]["speaker_id"].values
            inner_val_speakers = inner_speakers.iloc[inner_val_idx]["speaker_id"].values
            
            inner_train = outer_train[outer_train["speaker_id"].isin(inner_train_speakers)].reset_index(drop=True)
            inner_val = outer_train[outer_train["speaker_id"].isin(inner_val_speakers)].reset_index(drop=True)
            inner_train_audio = outer_train_audio[outer_train_audio["speaker_id"].isin(inner_train_speakers)].reset_index(drop=True)
            inner_val_audio = outer_train_audio[outer_train_audio["speaker_id"].isin(inner_val_speakers)].reset_index(drop=True)
            
            text_model = make_meta_model(args)
            audio_model = make_meta_model(args)
            text_model.fit(inner_train[feature_cols_text].to_numpy(), inner_train["y_true"].to_numpy())
            audio_model.fit(inner_train_audio[audio_feature_cols].to_numpy(), inner_train_audio["y_true"].to_numpy())
            
            if args.task == "classification":
                text_probs = text_model.predict_proba(inner_val[feature_cols_text].to_numpy())
                audio_probs = audio_model.predict_proba(inner_val_audio[audio_feature_cols].to_numpy())
                speaker_to_idx = {sp: i for i, sp in enumerate(outer_train["speaker_id"])}
                for i, row in inner_val.iterrows():
                    sp = row["speaker_id"]
                    outer_idx = speaker_to_idx[sp]
                    oof_text_probs[outer_idx] = text_probs[i]
                    oof_audio_probs[outer_idx] = audio_probs[i]
            else:
                text_preds = text_model.predict(inner_val[feature_cols_text].to_numpy())
                audio_preds = audio_model.predict(inner_val_audio[audio_feature_cols].to_numpy())
                speaker_to_idx = {sp: i for i, sp in enumerate(outer_train["speaker_id"])}
                for i, row in inner_val.iterrows():
                    sp = row["speaker_id"]
                    outer_idx = speaker_to_idx[sp]
                    oof_text_preds[outer_idx] = text_preds[i]
                    oof_audio_preds[outer_idx] = audio_preds[i]
        
        # Build meta-model on OOF predictions
        if args.task == "classification":
            X_meta_train = np.concatenate([oof_text_probs, oof_audio_probs], axis=1)
        else:
            X_meta_train = np.column_stack([oof_text_preds, oof_audio_preds])
        y_meta_train = outer_train["y_true"].values
        
        if args.task == "classification":
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=args.seed))
            ])
        else:
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        meta_model.fit(X_meta_train, y_meta_train)
        
        # Train final base models on outer_train
        text_result = train_meta_model_cv(
            outer_train, outer_val, None, feature_cols_text, args, 
            out_dir / f"stack_text_fold{fold_idx}"
        )
        audio_result = train_meta_model_cv(
            outer_train_audio, outer_val_audio, None, audio_feature_cols, args, 
            out_dir / f"stack_audio_fold{fold_idx}"
        )
        text_final = joblib.load(out_dir / f"stack_text_fold{fold_idx}" / "final_cv_model.joblib")
        audio_final = joblib.load(out_dir / f"stack_audio_fold{fold_idx}" / "final_cv_model.joblib")
        text_selected = pd.read_csv(out_dir / f"stack_text_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        audio_selected = pd.read_csv(out_dir / f"stack_audio_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        X_text_val = outer_val[text_selected].to_numpy()
        X_audio_val = outer_val_audio[audio_selected].to_numpy()
        
        if args.task == "classification":
            text_probs_val = text_final.predict_proba(X_text_val)
            audio_probs_val = audio_final.predict_proba(X_audio_val)
            X_meta_val = np.concatenate([text_probs_val, audio_probs_val], axis=1)
            preds = meta_model.predict(X_meta_val)
        else:
            text_preds_val = text_final.predict(X_text_val)
            audio_preds_val = audio_final.predict(X_audio_val)
            X_meta_val = np.column_stack([text_preds_val, audio_preds_val])
            preds = meta_model.predict(X_meta_val)
        
        fold_metrics.append(score_meta_model(None, preds, outer_val["y_true"].values, args.task))
        all_preds.extend(preds)
        all_true.extend(outer_val["y_true"].values)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "model_based_fusion_fold_metrics.csv", index=False)
    
    pred_df = all_text_data[["speaker_id", "y_true"]].copy()
    pred_df["y_pred"] = all_preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    return aggregate_metrics, all_preds, all_true


# =======================================================================
#  FIXED: MAIN FUSION ORCHESTRATOR - Correct order and all functions
# =======================================================================

def run_fusion_experiments(
    train_features, val_features, test_features,
    feature_cols_text, audio_df, args, out_dir: Path,
    text_meta_model=None, text_selected_features=None
):
    """
    Run ALL fusion experiments with audio features properly used.
    EXITS if audio-only fails.
    """
    fusion_out = out_dir / "fusion_results"
    fusion_out.mkdir(parents=True, exist_ok=True)

    # Get audio feature columns
    audio_feature_cols = get_audio_feature_cols(audio_df)
    
    print(f"\n{'='*60}")
    print(f"RUNNING ALL FUSION EXPERIMENTS")
    print(f"  Text features: {len(feature_cols_text)}")
    print(f"  Audio features: {len(audio_feature_cols)}")
    print(f"  Audio columns: {audio_feature_cols[:5]}...")
    print(f"  Results saved to: {fusion_out}")
    print(f"{'='*60}")

    # ============================================================
    # CRITICAL: Merge audio features with ALL splits
    # ============================================================
    print("\nMerging audio features with text features...")
    train_with_audio = merge_audio_features(train_features, audio_df, how='left')
    val_with_audio = merge_audio_features(val_features, audio_df, how='left')
    
    if test_features is not None and not test_features.empty:
        test_with_audio = merge_audio_features(test_features, audio_df, how='left')
    else:
        test_with_audio = None
    
    print(f"  Train: {len(train_with_audio)} samples, {len(train_with_audio.columns)} columns")
    print(f"  Val: {len(val_with_audio)} samples, {len(val_with_audio.columns)} columns")
    if test_with_audio is not None:
        print(f"  Test: {len(test_with_audio)} samples, {len(test_with_audio.columns)} columns")
    
    # Verify audio features are present
    audio_cols_present = [c for c in audio_feature_cols if c in train_with_audio.columns]
    print(f"  Audio columns present in merged data: {len(audio_cols_present)}/{len(audio_feature_cols)}")
    
    # ============================================================
    # CRITICAL CHECK: Must have audio features to continue
    # ============================================================
    if len(audio_cols_present) == 0:
        error_msg = (
            "\n" + "="*60 + "\n"
            "❌ CRITICAL ERROR: No audio features found in merged data!\n"
            f"   Audio columns requested: {audio_feature_cols[:5]}...\n"
            f"   Columns available in train data: {train_with_audio.columns.tolist()[:10]}...\n"
            "   Please check:\n"
            "     1. Audio CSV path is correct\n"
            "     2. Speaker IDs match between text and audio data\n"
            "     3. Audio CSV has numeric feature columns\n"
            "     4. --audio-feature-cols (if provided) exist in the CSV\n"
            "="*60
        )
        print(error_msg)
        raise ValueError(error_msg)

    results = {}

    # ============================================================
    # 1. AUDIO ONLY - Runs FIRST - MUST SUCCEED
    # ============================================================
    print("\n" + "="*60)
    print("1. AUDIO ONLY (audio features only - NO TEXT) - RUNNING FIRST")
    print("="*60)
    
    audio_result = None
    try:
        audio_result = run_audio_only_baseline(
            train_with_audio, val_with_audio, test_with_audio,
            audio_feature_cols, args, fusion_out
        )
        
        # Check if audio-only succeeded
        if audio_result is None:
            error_msg = (
                "\n" + "="*60 + "\n"
                "❌ AUDIO-ONLY FAILED - Stopping all experiments!\n"
                "   Audio-only returned None. This could be due to:\n"
                "     1. Not enough training samples\n"
                "     2. Training failed during cross-validation\n"
                "     3. Numerical issues with audio features\n"
                "     4. All folds failed\n"
                "   Check the logs above for specific errors.\n"
                "="*60
            )
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Check if audio_result has valid metrics
        if args.task == "classification":
            if isinstance(audio_result, dict):
                if "macro_f1" not in audio_result and "test_metrics" not in audio_result:
                    error_msg = (
                        "\n" + "="*60 + "\n"
                        "❌ AUDIO-ONLY FAILED - Invalid result structure!\n"
                        f"   Expected 'macro_f1' or 'test_metrics' in result\n"
                        f"   Got: {audio_result.keys() if isinstance(audio_result, dict) else type(audio_result)}\n"
                        "="*60
                    )
                    print(error_msg)
                    raise ValueError(error_msg)
        else:
            if isinstance(audio_result, dict):
                if "rmse" not in audio_result and "test_metrics" not in audio_result:
                    error_msg = (
                        "\n" + "="*60 + "\n"
                        "❌ AUDIO-ONLY FAILED - Invalid result structure!\n"
                        f"   Expected 'rmse' or 'test_metrics' in result\n"
                        f"   Got: {audio_result.keys() if isinstance(audio_result, dict) else type(audio_result)}\n"
                        "="*60
                    )
                    print(error_msg)
                    raise ValueError(error_msg)
        
        print("  ✅ Audio-only completed successfully")
        results['audio_only'] = audio_result
        
    except Exception as e:
        error_msg = (
            "\n" + "="*60 + "\n"
            f"❌ AUDIO-ONLY FAILED with exception: {e}\n"
            "   Stopping all fusion experiments!\n"
            "   Please fix the audio feature issues and re-run.\n"
            "="*60
        )
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg) from e

    # ============================================================
    # 2. TEXT ONLY - Runs SECOND (only if audio succeeded)
    # ============================================================
    print("\n" + "="*60)
    print("2. TEXT ONLY (reusing main pipeline model - NO AUDIO) - RUNNING SECOND")
    print("="*60)
    
    try:
        results['text_only'] = run_text_only_baseline(
            train_features, val_features, test_features,
            feature_cols_text, args, fusion_out,
            text_meta_model, text_selected_features
        )
        print("  ✅ Text-only completed successfully")
    except Exception as e:
        print(f"  ❌ Text-only crashed: {e}")
        # Don't stop on text-only failure, but log it
        results['text_only'] = None

    # ============================================================
    # 3. EARLY FUSION - Concatenates text + audio features
    # ============================================================
    print("\n" + "="*60)
    print("3. EARLY FUSION (concat text + audio features)")
    print("="*60)
    
    try:
        results['early_fusion'] = run_early_fusion(
            train_with_audio, val_with_audio, test_with_audio,
            feature_cols_text, audio_feature_cols, args, fusion_out
        )
        print("  ✅ Early fusion completed successfully")
    except Exception as e:
        print(f"  ❌ Early fusion crashed: {e}")
        results['early_fusion'] = None

    # ============================================================
    # 4. LATE FUSION - Separate text and audio models
    # ============================================================
    print("\n" + "="*60)
    print("4. LATE FUSION (separate models, average predictions)")
    print("="*60)
    
    try:
        results['late_fusion'] = run_late_fusion(
            train_features, val_features, test_features,
            train_with_audio, val_with_audio, test_with_audio,
            feature_cols_text, audio_feature_cols, args, fusion_out,
            text_meta_model, text_selected_features
        )
        print("  ✅ Late fusion completed successfully")
    except Exception as e:
        print(f"  ❌ Late fusion crashed: {e}")
        results['late_fusion'] = None

    # ============================================================
    # 5. MODEL-BASED FUSION - Stacking
    # ============================================================
    print("\n" + "="*60)
    print("5. MODEL-BASED FUSION (stacking)")
    print("="*60)
    
    try:
        results['model_based_fusion'] = run_model_based_fusion(
            train_features, val_features, test_features,
            train_with_audio, val_with_audio, test_with_audio,
            feature_cols_text, audio_feature_cols, args, fusion_out,
            text_meta_model, text_selected_features
        )
        print("  ✅ Model-based fusion completed successfully")
    except Exception as e:
        print(f"  ❌ Model-based fusion crashed: {e}")
        results['model_based_fusion'] = None

    # ============================================================
    # 6. CONFIDENCE-WEIGHTED FUSION (Novel)
    # ============================================================
    if getattr(args, 'fusion_novel', 'none') in ['confidence', 'all']:
        print("\n" + "="*60)
        print("6. CONFIDENCE-WEIGHTED LATE FUSION (Novel)")
        print("="*60)
        
        try:
            results['confidence_weighted'] = run_confidence_weighted_fusion(
                train_features, val_features, test_features,
                train_with_audio, val_with_audio, test_with_audio,
                feature_cols_text, audio_feature_cols, args, fusion_out,
                text_meta_model, text_selected_features
            )
            print("  ✅ Confidence-weighted fusion completed successfully")
        except Exception as e:
            print(f"  ❌ Confidence-weighted fusion crashed: {e}")
            results['confidence_weighted'] = None

    # ============================================================
    # 7. INTERACTION STACKING (Novel)
    # ============================================================
    if getattr(args, 'fusion_novel', 'none') in ['interaction', 'all']:
        print("\n" + "="*60)
        print("7. INTERACTION STACKING (Novel)")
        print("="*60)
        
        try:
            results['interaction_stacking'] = run_interaction_stacking_fusion(
                train_features, val_features, test_features,
                train_with_audio, val_with_audio, test_with_audio,
                feature_cols_text, audio_feature_cols, args, fusion_out,
                text_meta_model, text_selected_features
            )
            print("  ✅ Interaction stacking completed successfully")
        except Exception as e:
            print(f"  ❌ Interaction stacking crashed: {e}")
            results['interaction_stacking'] = None

    # ============================================================
    # 8. MIXTURE OF EXPERTS (Novel)
    # ============================================================
    if getattr(args, 'fusion_novel', 'none') in ['moe', 'all']:
        print("\n" + "="*60)
        print("8. MIXTURE OF EXPERTS (Novel)")
        print("="*60)
        
        try:
            results['mixture_of_experts'] = run_mixture_of_experts_fusion(
                train_features, val_features, test_features,
                train_with_audio, val_with_audio, test_with_audio,
                feature_cols_text, audio_feature_cols, args, fusion_out
            )
            print("  ✅ Mixture of experts completed successfully")
        except Exception as e:
            print(f"  ❌ Mixture of experts crashed: {e}")
            results['mixture_of_experts'] = None

    # ============================================================
    # 9. MLP EARLY FUSION (Novel)
    # ============================================================
    if getattr(args, 'fusion_novel', 'none') in ['mlp', 'all']:
        print("\n" + "="*60)
        print("9. MLP EARLY FUSION (Novel)")
        print("="*60)
        
        try:
            results['mlp_early_fusion'] = run_mlp_early_fusion_fusion(
                train_features, val_features, test_features,
                train_with_audio, val_with_audio, test_with_audio,
                feature_cols_text, audio_feature_cols, args, fusion_out
            )
            print("  ✅ MLP early fusion completed successfully")
        except Exception as e:
            print(f"  ❌ MLP early fusion crashed: {e}")
            results['mlp_early_fusion'] = None

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("FUSION EXPERIMENTS SUMMARY")
    print("="*60)
    
    summary_rows = []
    for name, result in results.items():
        if result is None:
            summary_rows.append({
                "fusion_method": name,
                "primary_score": "N/A",
                "status": "❌ FAILED or SKIPPED"
            })
        else:
            # Extract primary score
            if args.task == "classification":
                if isinstance(result, dict):
                    if "test_metrics" in result and isinstance(result["test_metrics"], dict):
                        score = result["test_metrics"].get("macro_f1", 0.0)
                    elif "macro_f1" in result:
                        score = result.get("macro_f1", 0.0)
                    elif "cv_aggregate_metrics" in result and isinstance(result["cv_aggregate_metrics"], dict):
                        score = result["cv_aggregate_metrics"].get("macro_f1", 0.0)
                    elif "aggregate_metrics" in result and isinstance(result["aggregate_metrics"], dict):
                        score = result["aggregate_metrics"].get("macro_f1", 0.0)
                    else:
                        score = 0.0
                else:
                    score = 0.0
            else:
                if isinstance(result, dict):
                    if "test_metrics" in result and isinstance(result["test_metrics"], dict):
                        score = -result["test_metrics"].get("rmse", float('inf'))
                    elif "rmse" in result:
                        score = -result.get("rmse", float('inf'))
                    elif "cv_aggregate_metrics" in result and isinstance(result["cv_aggregate_metrics"], dict):
                        score = -result["cv_aggregate_metrics"].get("rmse", float('inf'))
                    else:
                        score = -float('inf')
                else:
                    score = -float('inf')
            
            summary_rows.append({
                "fusion_method": name,
                "primary_score": round(score, 4) if score != float('inf') and score != -float('inf') else "N/A",
                "status": "✅ SUCCESS"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("primary_score", ascending=False)
    summary_df.to_csv(fusion_out / "fusion_summary.csv", index=False)
    
    print("\n" + summary_df.to_string(index=False))
    
    # Find best method
    if not summary_df.empty:
        # Filter out failed methods for best method
        success_df = summary_df[summary_df['status'] == "✅ SUCCESS"]
        if not success_df.empty:
            best_row = success_df.iloc[0]
            print(f"\n🏆 BEST FUSION METHOD: {best_row['fusion_method']} (score: {best_row['primary_score']:.4f})")
    
    return results


# =======================================================================
#  FIXED: EARLY FUSION - Properly uses audio features
# =======================================================================

def run_early_fusion(train_features, val_features, test_features,
                     feature_cols_text, audio_feature_cols, args, out_dir: Path):
    """
    EARLY FUSION - concatenates text and audio features.
    NOTE: train_features, val_features, test_features should already have audio merged.
    """
    early_dir = out_dir / "early_fusion"
    early_dir.mkdir(parents=True, exist_ok=True)
    
    early_feature_cols = feature_cols_text + audio_feature_cols
    print(f"  Early fusion features: {len(early_feature_cols)} total")
    print(f"    Text: {len(feature_cols_text)}, Audio: {len(audio_feature_cols)}")
    
    if args.test_frac == 0:
        result = train_meta_model_cv(
            train_features, val_features, test_features, 
            early_feature_cols, args, early_dir
        )
    else:
        result = train_meta_model_with_cv_selection(
            train_features, val_features, test_features, 
            early_feature_cols, args, early_dir
        )
    
    with open(early_dir / "fusion_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


# =======================================================================
#  MISSING FUNCTION: run_confidence_weighted_fusion
# =======================================================================

def run_confidence_weighted_fusion(train_features, val_features, test_features,
                                   train_with_audio, val_with_audio, test_with_audio,
                                   feature_cols_text, audio_feature_cols, args, out_dir: Path,
                                   text_meta_model=None, text_selected_features=None):
    """
    CONFIDENCE-WEIGHTED LATE FUSION - weights by inverse entropy.
    Uses audio features properly.
    """
    cw_dir = out_dir / "confidence_weighted_fusion"
    cw_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_frac == 0:
        metrics, preds, y_true = confidence_weighted_late_fusion_cv(
            train_features, val_features,
            train_with_audio, val_with_audio,
            feature_cols_text, audio_feature_cols, args, cw_dir
        )
        with open(cw_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        # Text model
        if text_meta_model is not None and text_selected_features is not None:
            text_model = text_meta_model
            text_selected = text_selected_features
        else:
            text_result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features,
                feature_cols_text, args, cw_dir / "text_model"
            )
            text_model = joblib.load(cw_dir / "text_model" / "meta_model.joblib")
            text_selected = pd.read_csv(cw_dir / "text_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Audio model
        audio_result = train_meta_model_with_cv_selection(
            train_with_audio, val_with_audio, test_with_audio,
            audio_feature_cols, args, cw_dir / "audio_model"
        )
        audio_model = joblib.load(cw_dir / "audio_model" / "meta_model.joblib")
        audio_selected = pd.read_csv(cw_dir / "audio_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        X_text_test = test_features[text_selected].to_numpy()
        X_audio_test = test_with_audio[audio_selected].to_numpy()
        y_true = test_features["y_true"].to_numpy()
        
        if args.task == "classification":
            prob_text = text_model.predict_proba(X_text_test)
            prob_audio = audio_model.predict_proba(X_audio_test)
            
            eps = 1e-12
            entropy_text = -np.sum(prob_text * np.log(prob_text + eps), axis=1)
            entropy_audio = -np.sum(prob_audio * np.log(prob_audio + eps), axis=1)
            
            w_text = 1.0 / (entropy_text + eps)
            w_audio = 1.0 / (entropy_audio + eps)
            total = w_text + w_audio
            w_text /= total
            w_audio /= total
            
            fused_probs = prob_text * w_text[:, None] + prob_audio * w_audio[:, None]
            preds = np.argmax(fused_probs, axis=1)
        else:
            pred_text = text_model.predict(X_text_test)
            pred_audio = audio_model.predict(X_audio_test)
            preds = (pred_text + pred_audio) / 2.0
        
        metrics = score_meta_model(None, preds, y_true, args.task)
        
        with open(cw_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = preds
        pred_df.to_csv(cw_dir / "predictions.csv", index=False)
        
        return metrics


# =======================================================================
#  MISSING FUNCTION: confidence_weighted_late_fusion_cv
# =======================================================================

def confidence_weighted_late_fusion_cv(train_features, val_features,
                                       train_with_audio, val_with_audio,
                                       feature_cols_text, audio_feature_cols, args, out_dir):
    """Confidence-weighted late fusion using CV with audio features."""
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
    all_preds = []
    all_true = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_speakers = speakers.iloc[train_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_idx]["speaker_id"].values
        
        fold_train = all_text_data[all_text_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val = all_text_data[all_text_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        fold_train_audio = all_audio_data[all_audio_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val_audio = all_audio_data[all_audio_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Text model
        text_result = train_meta_model_cv(
            fold_train, fold_val, None, feature_cols_text, args, 
            out_dir / f"cw_text_fold{fold_idx}"
        )
        text_model = joblib.load(out_dir / f"cw_text_fold{fold_idx}" / "final_cv_model.joblib")
        text_selected = pd.read_csv(out_dir / f"cw_text_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        # Audio model
        audio_result = train_meta_model_cv(
            fold_train_audio, fold_val_audio, None, audio_feature_cols, args, 
            out_dir / f"cw_audio_fold{fold_idx}"
        )
        audio_model = joblib.load(out_dir / f"cw_audio_fold{fold_idx}" / "final_cv_model.joblib")
        audio_selected = pd.read_csv(out_dir / f"cw_audio_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        X_text_val = fold_val[text_selected].to_numpy()
        X_audio_val = fold_val_audio[audio_selected].to_numpy()
        y_val = fold_val["y_true"].to_numpy()
        
        if args.task == "classification":
            prob_text = text_model.predict_proba(X_text_val)
            prob_audio = audio_model.predict_proba(X_audio_val)
            
            eps = 1e-12
            entropy_text = -np.sum(prob_text * np.log(prob_text + eps), axis=1)
            entropy_audio = -np.sum(prob_audio * np.log(prob_audio + eps), axis=1)
            
            w_text = 1.0 / (entropy_text + eps)
            w_audio = 1.0 / (entropy_audio + eps)
            total = w_text + w_audio
            w_text /= total
            w_audio /= total
            
            fused_probs = prob_text * w_text[:, None] + prob_audio * w_audio[:, None]
            preds = np.argmax(fused_probs, axis=1)
        else:
            pred_text = text_model.predict(X_text_val)
            pred_audio = audio_model.predict(X_audio_val)
            preds = (pred_text + pred_audio) / 2.0
        
        fold_metrics.append(score_meta_model(None, preds, y_val, args.task))
        all_preds.extend(preds)
        all_true.extend(y_val)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "confidence_weighted_fold_metrics.csv", index=False)
    
    pred_df = all_text_data[["speaker_id", "y_true"]].copy()
    pred_df["y_pred"] = all_preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    return aggregate_metrics, all_preds, all_true


# =======================================================================
#  MISSING FUNCTION: run_interaction_stacking_fusion
# =======================================================================

def run_interaction_stacking_fusion(train_features, val_features, test_features,
                                    train_with_audio, val_with_audio, test_with_audio,
                                    feature_cols_text, audio_feature_cols, args, out_dir: Path,
                                    text_meta_model=None, text_selected_features=None):
    """
    INTERACTION STACKING - stacking with cross-modal interaction features.
    Uses audio features properly.
    """
    inter_dir = out_dir / "interaction_stacking"
    inter_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_frac == 0:
        metrics, preds, y_true = interaction_stacking_cv(
            train_features, val_features,
            train_with_audio, val_with_audio,
            feature_cols_text, audio_feature_cols, args, inter_dir
        )
        with open(inter_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        # Text model
        if text_meta_model is not None and text_selected_features is not None:
            text_model = text_meta_model
            text_selected = text_selected_features
        else:
            text_result = train_meta_model_with_cv_selection(
                train_features, val_features, test_features,
                feature_cols_text, args, inter_dir / "text_model"
            )
            text_model = joblib.load(inter_dir / "text_model" / "meta_model.joblib")
            text_selected = pd.read_csv(inter_dir / "text_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Audio model
        audio_result = train_meta_model_with_cv_selection(
            train_with_audio, val_with_audio, test_with_audio,
            audio_feature_cols, args, inter_dir / "audio_model"
        )
        audio_model = joblib.load(inter_dir / "audio_model" / "meta_model.joblib")
        audio_selected = pd.read_csv(inter_dir / "audio_model" / "selected_embedding_features.csv")["feature"].tolist()
        
        # Get predictions with interactions
        train_text_X = train_features[text_selected].to_numpy()
        train_audio_X = train_with_audio[audio_selected].to_numpy()
        val_text_X = val_features[text_selected].to_numpy()
        val_audio_X = val_with_audio[audio_selected].to_numpy()
        test_text_X = test_features[text_selected].to_numpy()
        test_audio_X = test_with_audio[audio_selected].to_numpy()
        
        if args.task == "classification":
            classes = np.unique(train_features["y_true"])
            n_classes = len(classes)
            
            def get_probs(model, X):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    preds = model.predict(X)
                    return np.eye(n_classes)[preds.astype(int)]
            
            train_probs_text = get_probs(text_model, train_text_X)
            train_probs_audio = get_probs(audio_model, train_audio_X)
            val_probs_text = get_probs(text_model, val_text_X)
            val_probs_audio = get_probs(audio_model, val_audio_X)
            test_probs_text = get_probs(text_model, test_text_X)
            test_probs_audio = get_probs(audio_model, test_audio_X)
            
            X_train_meta = np.concatenate([
                train_probs_text, train_probs_audio,
                train_probs_text * train_probs_audio,
                np.abs(train_probs_text - train_probs_audio),
                train_probs_text ** 2,
                train_probs_audio ** 2
            ], axis=1)
            
            X_val_meta = np.concatenate([
                val_probs_text, val_probs_audio,
                val_probs_text * val_probs_audio,
                np.abs(val_probs_text - val_probs_audio),
                val_probs_text ** 2,
                val_probs_audio ** 2
            ], axis=1)
            
            X_test_meta = np.concatenate([
                test_probs_text, test_probs_audio,
                test_probs_text * test_probs_audio,
                np.abs(test_probs_text - test_probs_audio),
                test_probs_text ** 2,
                test_probs_audio ** 2
            ], axis=1)
        else:
            train_preds_text = text_model.predict(train_text_X)
            train_preds_audio = audio_model.predict(train_audio_X)
            val_preds_text = text_model.predict(val_text_X)
            val_preds_audio = audio_model.predict(val_audio_X)
            test_preds_text = text_model.predict(test_text_X)
            test_preds_audio = audio_model.predict(test_audio_X)
            
            X_train_meta = np.concatenate([
                train_preds_text.reshape(-1, 1), train_preds_audio.reshape(-1, 1),
                (train_preds_text * train_preds_audio).reshape(-1, 1),
                np.abs(train_preds_text - train_preds_audio).reshape(-1, 1),
                (train_preds_text ** 2).reshape(-1, 1),
                (train_preds_audio ** 2).reshape(-1, 1)
            ], axis=1)
            
            X_val_meta = np.concatenate([
                val_preds_text.reshape(-1, 1), val_preds_audio.reshape(-1, 1),
                (val_preds_text * val_preds_audio).reshape(-1, 1),
                np.abs(val_preds_text - val_preds_audio).reshape(-1, 1),
                (val_preds_text ** 2).reshape(-1, 1),
                (val_preds_audio ** 2).reshape(-1, 1)
            ], axis=1)
            
            X_test_meta = np.concatenate([
                test_preds_text.reshape(-1, 1), test_preds_audio.reshape(-1, 1),
                (test_preds_text * test_preds_audio).reshape(-1, 1),
                np.abs(test_preds_text - test_preds_audio).reshape(-1, 1),
                (test_preds_text ** 2).reshape(-1, 1),
                (test_preds_audio ** 2).reshape(-1, 1)
            ], axis=1)
        
        y_train = train_features["y_true"].to_numpy()
        y_val = val_features["y_true"].to_numpy()
        y_test = test_features["y_true"].to_numpy()
        
        X_meta_all = np.concatenate([X_train_meta, X_val_meta], axis=0)
        y_meta_all = np.concatenate([y_train, y_val], axis=0)
        
        if args.task == "classification":
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=args.seed))
            ])
        else:
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        
        meta_model.fit(X_meta_all, y_meta_all)
        y_pred = meta_model.predict(X_test_meta)
        metrics = score_meta_model(None, y_pred, y_test, args.task)
        
        with open(inter_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = y_pred
        pred_df.to_csv(inter_dir / "predictions.csv", index=False)
        
        return metrics


# =======================================================================
#  MISSING FUNCTION: interaction_stacking_cv
# =======================================================================

def interaction_stacking_cv(train_features, val_features,
                            train_with_audio, val_with_audio,
                            feature_cols_text, audio_feature_cols, args, out_dir):
    """Interaction stacking using CV with audio features."""
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv_outer = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(cv_outer.split(speakers, speakers["label"]))
    else:
        cv_outer = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        outer_splits = list(cv_outer.split(speakers))
    
    all_preds = []
    all_true = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
        train_speakers = speakers.iloc[train_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_idx]["speaker_id"].values
        
        outer_train = all_text_data[all_text_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        outer_val = all_text_data[all_text_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        outer_train_audio = all_audio_data[all_audio_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        outer_val_audio = all_audio_data[all_audio_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Inner CV for OOF predictions
        inner_folds = min(3, len(outer_train) // 10) if len(outer_train) > 10 else 2
        inner_folds = max(2, inner_folds)
        n_train = len(outer_train)
        
        if args.task == "classification":
            classes = np.unique(outer_train["y_true"])
            n_classes = len(classes)
            oof_text_probs = np.zeros((n_train, n_classes))
            oof_audio_probs = np.zeros((n_train, n_classes))
        else:
            oof_text_preds = np.zeros(n_train)
            oof_audio_preds = np.zeros(n_train)
        
        inner_speakers = outer_train.groupby("speaker_id")["y_true"].first().reset_index()
        inner_speakers.columns = ["speaker_id", "label"]
        
        if args.task == "classification":
            cv_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=args.seed+fold_idx)
            inner_splits = list(cv_inner.split(inner_speakers, inner_speakers["label"]))
        else:
            cv_inner = KFold(n_splits=inner_folds, shuffle=True, random_state=args.seed+fold_idx)
            inner_splits = list(cv_inner.split(inner_speakers))
        
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_train_speakers = inner_speakers.iloc[inner_train_idx]["speaker_id"].values
            inner_val_speakers = inner_speakers.iloc[inner_val_idx]["speaker_id"].values
            
            inner_train = outer_train[outer_train["speaker_id"].isin(inner_train_speakers)].reset_index(drop=True)
            inner_val = outer_train[outer_train["speaker_id"].isin(inner_val_speakers)].reset_index(drop=True)
            inner_train_audio = outer_train_audio[outer_train_audio["speaker_id"].isin(inner_train_speakers)].reset_index(drop=True)
            inner_val_audio = outer_train_audio[outer_train_audio["speaker_id"].isin(inner_val_speakers)].reset_index(drop=True)
            
            text_model = make_meta_model(args)
            audio_model = make_meta_model(args)
            text_model.fit(inner_train[feature_cols_text].to_numpy(), inner_train["y_true"].to_numpy())
            audio_model.fit(inner_train_audio[audio_feature_cols].to_numpy(), inner_train_audio["y_true"].to_numpy())
            
            if args.task == "classification":
                text_probs = text_model.predict_proba(inner_val[feature_cols_text].to_numpy())
                audio_probs = audio_model.predict_proba(inner_val_audio[audio_feature_cols].to_numpy())
                speaker_to_idx = {sp: i for i, sp in enumerate(outer_train["speaker_id"])}
                for i, row in inner_val.iterrows():
                    sp = row["speaker_id"]
                    outer_idx = speaker_to_idx[sp]
                    oof_text_probs[outer_idx] = text_probs[i]
                    oof_audio_probs[outer_idx] = audio_probs[i]
            else:
                text_preds = text_model.predict(inner_val[feature_cols_text].to_numpy())
                audio_preds = audio_model.predict(inner_val_audio[audio_feature_cols].to_numpy())
                speaker_to_idx = {sp: i for i, sp in enumerate(outer_train["speaker_id"])}
                for i, row in inner_val.iterrows():
                    sp = row["speaker_id"]
                    outer_idx = speaker_to_idx[sp]
                    oof_text_preds[outer_idx] = text_preds[i]
                    oof_audio_preds[outer_idx] = audio_preds[i]
        
        # Build meta-features with interactions
        if args.task == "classification":
            X_meta_train = np.concatenate([
                oof_text_probs, oof_audio_probs,
                oof_text_probs * oof_audio_probs,
                np.abs(oof_text_probs - oof_audio_probs),
                oof_text_probs ** 2,
                oof_audio_probs ** 2
            ], axis=1)
        else:
            X_meta_train = np.concatenate([
                oof_text_preds.reshape(-1, 1), oof_audio_preds.reshape(-1, 1),
                (oof_text_preds * oof_audio_preds).reshape(-1, 1),
                np.abs(oof_text_preds - oof_audio_preds).reshape(-1, 1),
                (oof_text_preds ** 2).reshape(-1, 1),
                (oof_audio_preds ** 2).reshape(-1, 1)
            ], axis=1)
        
        y_meta_train = outer_train["y_true"].values
        
        if args.task == "classification":
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=args.seed))
            ])
        else:
            meta_model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        meta_model.fit(X_meta_train, y_meta_train)
        
        # Train final base models
        text_result = train_meta_model_cv(
            outer_train, outer_val, None, feature_cols_text, args, 
            out_dir / f"inter_text_fold{fold_idx}"
        )
        audio_result = train_meta_model_cv(
            outer_train_audio, outer_val_audio, None, audio_feature_cols, args, 
            out_dir / f"inter_audio_fold{fold_idx}"
        )
        text_final = joblib.load(out_dir / f"inter_text_fold{fold_idx}" / "final_cv_model.joblib")
        audio_final = joblib.load(out_dir / f"inter_audio_fold{fold_idx}" / "final_cv_model.joblib")
        text_selected = pd.read_csv(out_dir / f"inter_text_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        audio_selected = pd.read_csv(out_dir / f"inter_audio_fold{fold_idx}" / "final_selected_features.csv")["feature"].tolist()
        
        X_text_val = outer_val[text_selected].to_numpy()
        X_audio_val = outer_val_audio[audio_selected].to_numpy()
        
        if args.task == "classification":
            text_probs_val = text_final.predict_proba(X_text_val)
            audio_probs_val = audio_final.predict_proba(X_audio_val)
            X_meta_val = np.concatenate([
                text_probs_val, audio_probs_val,
                text_probs_val * audio_probs_val,
                np.abs(text_probs_val - audio_probs_val),
                text_probs_val ** 2,
                audio_probs_val ** 2
            ], axis=1)
        else:
            text_preds_val = text_final.predict(X_text_val)
            audio_preds_val = audio_final.predict(X_audio_val)
            X_meta_val = np.concatenate([
                text_preds_val.reshape(-1, 1), audio_preds_val.reshape(-1, 1),
                (text_preds_val * audio_preds_val).reshape(-1, 1),
                np.abs(text_preds_val - audio_preds_val).reshape(-1, 1),
                (text_preds_val ** 2).reshape(-1, 1),
                (audio_preds_val ** 2).reshape(-1, 1)
            ], axis=1)
        
        preds = meta_model.predict(X_meta_val)
        fold_metrics.append(score_meta_model(None, preds, outer_val["y_true"].values, args.task))
        all_preds.extend(preds)
        all_true.extend(outer_val["y_true"].values)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "interaction_stacking_fold_metrics.csv", index=False)
    
    pred_df = all_text_data[["speaker_id", "y_true"]].copy()
    pred_df["y_pred"] = all_preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    return aggregate_metrics, all_preds, all_true


# =======================================================================
#  MISSING FUNCTION: run_mixture_of_experts_fusion
# =======================================================================

def run_mixture_of_experts_fusion(train_features, val_features, test_features,
                                  train_with_audio, val_with_audio, test_with_audio,
                                  feature_cols_text, audio_feature_cols, args, out_dir: Path):
    """
    MIXTURE OF EXPERTS - clusters speakers, trains separate experts per cluster.
    Uses audio features properly.
    """
    moe_dir = out_dir / "mixture_of_experts"
    moe_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_frac == 0:
        metrics, preds, y_true = mixture_of_experts_cv(
            train_features, val_features,
            train_with_audio, val_with_audio,
            feature_cols_text, audio_feature_cols, args, moe_dir
        )
        with open(moe_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    else:
        # Combine text and audio for clustering
        all_text_data = pd.concat([train_features, val_features], ignore_index=True)
        all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
        
        # Get concatenated features for clustering (text + audio)
        concat_feats = np.concatenate([
            all_text_data[feature_cols_text].to_numpy(),
            all_audio_data[audio_feature_cols].to_numpy()
        ], axis=1)
        
        n_clusters = min(3, len(all_text_data) // 10) if len(all_text_data) > 30 else 2
        n_clusters = max(2, n_clusters)
        print(f"Mixture of Experts: using {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(concat_feats)
        all_text_data['cluster'] = cluster_labels
        all_audio_data['cluster'] = cluster_labels
        
        # Train gating model
        gate_model = LogisticRegression(multi_class='multinomial', random_state=args.seed, max_iter=500)
        gate_model.fit(concat_feats, cluster_labels)
        
        # Train experts per cluster (early fusion with text + audio)
        expert_models = {}
        for c in range(n_clusters):
            cluster_text = all_text_data[all_text_data['cluster'] == c]
            cluster_audio = all_audio_data[all_audio_data['cluster'] == c]
            if len(cluster_text) < 5:
                continue
            X_train = np.concatenate([
                cluster_text[feature_cols_text].to_numpy(),
                cluster_audio[audio_feature_cols].to_numpy()
            ], axis=1)
            y_train = cluster_text['y_true'].values
            model = make_meta_model(args)
            model.fit(X_train, y_train)
            expert_models[c] = model
        
        # Predict on test
        test_concat = np.concatenate([
            test_features[feature_cols_text].to_numpy(),
            test_with_audio[audio_feature_cols].to_numpy()
        ], axis=1)
        gate_probs = gate_model.predict_proba(test_concat)
        y_test = test_features["y_true"].to_numpy()
        n_test = len(test_features)
        
        if args.task == "classification":
            classes = np.unique(all_text_data['y_true'])
            n_classes = len(classes)
            pred_probs = np.zeros((n_test, n_classes))
            for c, model in expert_models.items():
                X_test = np.concatenate([
                    test_features[feature_cols_text].to_numpy(),
                    test_with_audio[audio_feature_cols].to_numpy()
                ], axis=1)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)
                    if list(model.classes_) != list(classes):
                        prob_map = np.zeros((n_test, n_classes))
                        for i, cls in enumerate(model.classes_):
                            if cls in classes:
                                idx_global = np.where(classes == cls)[0][0]
                                prob_map[:, idx_global] = probs[:, i]
                        probs = prob_map
                    pred_probs += gate_probs[:, c][:, None] * probs
                else:
                    preds = model.predict(X_test)
                    one_hot = np.eye(n_classes)[preds.astype(int)]
                    pred_probs += gate_probs[:, c][:, None] * one_hot
            preds = np.argmax(pred_probs, axis=1)
        else:
            preds = np.zeros(n_test)
            for c, model in expert_models.items():
                X_test = np.concatenate([
                    test_features[feature_cols_text].to_numpy(),
                    test_with_audio[audio_feature_cols].to_numpy()
                ], axis=1)
                preds += gate_probs[:, c] * model.predict(X_test)
        
        metrics = score_meta_model(None, preds, y_test, args.task)
        
        with open(moe_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = preds
        pred_df.to_csv(moe_dir / "predictions.csv", index=False)
        
        return metrics


# =======================================================================
#  MISSING FUNCTION: mixture_of_experts_cv
# =======================================================================

def mixture_of_experts_cv(train_features, val_features,
                          train_with_audio, val_with_audio,
                          feature_cols_text, audio_feature_cols, args, out_dir):
    """Mixture of Experts using CV with audio features."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    # Get concatenated features for clustering
    concat_feats = np.concatenate([
        all_text_data[feature_cols_text].to_numpy(),
        all_audio_data[audio_feature_cols].to_numpy()
    ], axis=1)
    
    n_clusters = min(3, len(all_text_data) // 10) if len(all_text_data) > 30 else 2
    n_clusters = max(2, n_clusters)
    print(f"Mixture of Experts: using {n_clusters} clusters")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(concat_feats)
    all_text_data['cluster'] = cluster_labels
    all_audio_data['cluster'] = cluster_labels
    
    speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
    speakers.columns = ["speaker_id", "label"]
    
    if args.task == "classification":
        cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers, speakers["label"]))
    else:
        cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(cv.split(speakers))
    
    all_preds = []
    all_true = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_speakers = speakers.iloc[train_idx]["speaker_id"].values
        val_speakers = speakers.iloc[val_idx]["speaker_id"].values
        
        fold_text = all_text_data[all_text_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val_text = all_text_data[all_text_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        fold_audio = all_audio_data[all_audio_data["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        fold_val_audio = all_audio_data[all_audio_data["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        
        # Train gating model on fold train
        train_concat = np.concatenate([
            fold_text[feature_cols_text].to_numpy(),
            fold_audio[audio_feature_cols].to_numpy()
        ], axis=1)
        gate_fold = LogisticRegression(multi_class='multinomial', random_state=args.seed, max_iter=500)
        gate_fold.fit(train_concat, fold_text['cluster'].values)
        
        # Train experts per cluster (early fusion with text + audio)
        expert_models = {}
        for c in range(n_clusters):
            cluster_text = fold_text[fold_text['cluster'] == c]
            cluster_audio = fold_audio[fold_audio['cluster'] == c]
            if len(cluster_text) < 5:
                continue
            X_train = np.concatenate([
                cluster_text[feature_cols_text].to_numpy(),
                cluster_audio[audio_feature_cols].to_numpy()
            ], axis=1)
            y_train = cluster_text['y_true'].values
            model = make_meta_model(args)
            model.fit(X_train, y_train)
            expert_models[c] = model
        
        # Predict on validation
        val_concat = np.concatenate([
            fold_val_text[feature_cols_text].to_numpy(),
            fold_val_audio[audio_feature_cols].to_numpy()
        ], axis=1)
        gate_probs = gate_fold.predict_proba(val_concat)
        y_val = fold_val_text['y_true'].values
        n_val = len(fold_val_text)
        
        if args.task == "classification":
            classes = np.unique(all_text_data['y_true'])
            n_classes = len(classes)
            pred_probs = np.zeros((n_val, n_classes))
            for c, model in expert_models.items():
                X_val = np.concatenate([
                    fold_val_text[feature_cols_text].to_numpy(),
                    fold_val_audio[audio_feature_cols].to_numpy()
                ], axis=1)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)
                    if list(model.classes_) != list(classes):
                        prob_map = np.zeros((n_val, n_classes))
                        for i, cls in enumerate(model.classes_):
                            if cls in classes:
                                idx_global = np.where(classes == cls)[0][0]
                                prob_map[:, idx_global] = probs[:, i]
                        probs = prob_map
                    pred_probs += gate_probs[:, c][:, None] * probs
                else:
                    preds = model.predict(X_val)
                    one_hot = np.eye(n_classes)[preds.astype(int)]
                    pred_probs += gate_probs[:, c][:, None] * one_hot
            preds = np.argmax(pred_probs, axis=1)
        else:
            preds = np.zeros(n_val)
            for c, model in expert_models.items():
                X_val = np.concatenate([
                    fold_val_text[feature_cols_text].to_numpy(),
                    fold_val_audio[audio_feature_cols].to_numpy()
                ], axis=1)
                preds += gate_probs[:, c] * model.predict(X_val)
        
        fold_metrics.append(score_meta_model(None, preds, y_val, args.task))
        all_preds.extend(preds)
        all_true.extend(y_val)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
    
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "mixture_of_experts_fold_metrics.csv", index=False)
    
    pred_df = all_text_data[["speaker_id", "y_true"]].copy()
    pred_df["y_pred"] = all_preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    return aggregate_metrics, all_preds, all_true


# =======================================================================
#  MISSING FUNCTION: run_mlp_early_fusion_fusion
# =======================================================================

def run_mlp_early_fusion_fusion(train_features, val_features, test_features,
                                train_with_audio, val_with_audio, test_with_audio,
                                feature_cols_text, audio_feature_cols, args, out_dir: Path):
    """
    MLP EARLY FUSION - uses MLP as meta-model on concatenated text+audio features.
    Uses audio features properly.
    """
    mlp_dir = out_dir / "mlp_early_fusion"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    
    # Combine text and audio data
    all_text_data = pd.concat([train_features, val_features], ignore_index=True)
    all_audio_data = pd.concat([train_with_audio, val_with_audio], ignore_index=True)
    
    # Concatenate text + audio features
    X_all = np.concatenate([
        all_text_data[feature_cols_text].to_numpy(),
        all_audio_data[audio_feature_cols].to_numpy()
    ], axis=1)
    y_all = all_text_data['y_true'].values
    
    if args.test_frac == 0:
        speakers = all_text_data.groupby("speaker_id")["y_true"].first().reset_index()
        speakers.columns = ["speaker_id", "label"]
        
        if args.task == "classification":
            cv = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
            fold_splits = list(cv.split(speakers, speakers["label"]))
        else:
            cv = KFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
            fold_splits = list(cv.split(speakers))
        
        all_preds = []
        all_true = []
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            train_speakers = speakers.iloc[train_idx]["speaker_id"].values
            val_speakers = speakers.iloc[val_idx]["speaker_id"].values
            
            train_mask = all_text_data["speaker_id"].isin(train_speakers)
            val_mask = all_text_data["speaker_id"].isin(val_speakers)
            
            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_val = X_all[val_mask]
            y_val = y_all[val_mask]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            if args.task == "classification":
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=500,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            else:
                mlp = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=500,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            mlp.fit(X_train_scaled, y_train)
            preds = mlp.predict(X_val_scaled)
            
            fold_metrics.append(score_meta_model(None, preds, y_val, args.task))
            all_preds.extend(preds)
            all_true.extend(y_val)
        
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        aggregate_metrics = score_meta_model(None, all_preds, all_true, args.task)
        
        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_csv(mlp_dir / "mlp_early_fusion_fold_metrics.csv", index=False)
        
        with open(mlp_dir / "fusion_metrics.json", "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        pred_df = all_text_data[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = all_preds
        pred_df.to_csv(mlp_dir / "predictions.csv", index=False)
        
        return aggregate_metrics
    else:
        # Held-out test mode
        X_train = X_all
        y_train = y_all
        X_test = np.concatenate([
            test_features[feature_cols_text].to_numpy(),
            test_with_audio[audio_feature_cols].to_numpy()
        ], axis=1)
        y_test = test_features["y_true"].to_numpy()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if args.task == "classification":
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=args.seed,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=args.seed,
                early_stopping=True,
                validation_fraction=0.1
            )
        mlp.fit(X_train_scaled, y_train)
        preds = mlp.predict(X_test_scaled)
        metrics = score_meta_model(None, preds, y_test, args.task)
        
        with open(mlp_dir / "fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        pred_df = test_features[["speaker_id", "y_true"]].copy()
        pred_df["y_pred"] = preds
        pred_df.to_csv(mlp_dir / "predictions.csv", index=False)
        
        return metrics
 
# =======================================================================
#  PARSER SETUP
# =======================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Train per‑question models with ensemble meta-model and audio fusion")
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
    parser.add_argument("--hpo-n-trials", type=int, default=30)
    parser.add_argument("--hpo-timeout", type=int, default=None)
    parser.add_argument("--hpo-folds", type=int, default=5)
    parser.add_argument("--force-hpo", action="store_true")
    
    # Meta-model settings
    parser.add_argument("--meta-model", 
                        choices=["linear", "random_forest", "svm", "hist_gradient_boosting", "gradient_boosting", "knn",
                                "ridge", "lasso", "elasticnet"],
                        default="linear")
    parser.add_argument("--use-ensemble", action="store_true")
    parser.add_argument("--ensemble-models", nargs="+",
                        choices=["linear", "random_forest", "svm", "hist_gradient_boosting", "gradient_boosting", "knn",
                                "ridge", "lasso", "elasticnet"],
                        default=["linear", "random_forest", "hist_gradient_boosting"])
    parser.add_argument("--ensemble-weights", nargs="+", type=float, default=None)
    
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
    
    # Audio features and fusion
    parser.add_argument("--audio-features-csv", type=str, default=None)
    parser.add_argument("--audio-feature-cols", nargs="+", default=None)
    parser.add_argument("--fusion-methods", nargs="+", 
                        choices=["early", "late", "model_based", "text_only", "audio_only", "all"],
                        default=["all"])
    parser.add_argument("--audio-meta-model", 
                        choices=["linear", "random_forest", "svm", "hist_gradient_boosting", "gradient_boosting", "knn"],
                        default="linear")
    
    # Novel fusion methods
    parser.add_argument("--fusion-novel", 
                        choices=["none", "confidence", "interaction", "moe", "mlp", "all"],
                        default="none",
                        help="Novel fusion approach to use in addition to baselines")
    
    return parser


# =======================================================================
#  MAIN
# =======================================================================

def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    # Remove stale files
    for f in out_dir.glob("*per_question_validation_scores.csv"):
        f.unlink()
        print(f"Removed stale score file: {f.name}")

    if args.audio_features_csv is not None:
        print("\nAudio features CSV provided. Running ALL fusion methods.")
        args.fusion_methods = ['all']

    # Check for existing results
    selected_questions_file = out_dir / "selected_questions.csv"
    cv_k_results_file = out_dir / "cv_k_selection_results.csv"
    test_metrics_file = out_dir / "meta_test_metrics.json"
    
    if selected_questions_file.exists() and cv_k_results_file.exists() and test_metrics_file.exists() and not args.force_hpo:
        print("\n" + "="*60)
        print("FINAL RESULTS ALREADY EXIST")
        print("="*60)
        print(f"Found existing results at:")
        print(f"  - {selected_questions_file}")
        print(f"  - {cv_k_results_file}")
        print(f"  - {test_metrics_file}")
        print(f"\nSkipping all processing.")
        print(f"To re-run, either:")
        print(f"  1. Delete {out_dir} directory, or")
        print(f"  2. Use --force-hpo flag")
        print("="*60 + "\n")
        
        with open(test_metrics_file, 'r') as f:
            existing_metrics = json.load(f)
        print("Existing results summary:")
        print(json.dumps(existing_metrics, indent=2))
        return
    
    # Full pipeline
    print("\n" + "="*60)
    print("STARTING FULL PIPELINE")
    print("="*60)
    
    cleanup_old_splits(splits_dir)
    (out_dir / "question_ensemble_config.json").write_text(json.dumps(vars(args), indent=2))
    
    # Load data
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
    
    # Hyperparameter search
    best_hparams_path = out_dir / "best_hyperparams_all_questions.json"
    if best_hparams_path.exists() and not args.force_hpo:
        best_hparams = json.loads(best_hparams_path.read_text())
        print(f"Loaded best hyperparameters from {best_hparams_path}")
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
    
    # Update args
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
    
    if args.test_frac > 0:
        test_features, _ = build_feature_table(embedding_files["test"], available_qs)
    else:
        print("test_frac=0: No test features will be created")
        test_features = pd.DataFrame(columns=train_features.columns) if train_features is not None else pd.DataFrame()
    
    # Save raw feature tables
    train_features.to_csv(out_dir / "meta_train_features.csv", index=False)
    val_features.to_csv(out_dir / "meta_val_features.csv", index=False)
    
    # Align feature columns
    if test_features is not None and not test_features.empty:
        train_features, val_features, test_features = align_feature_tables(
            train_features, val_features, test_features, feature_cols
        )
    else:
        train_features, val_features, _ = align_feature_tables(
            train_features, val_features, pd.DataFrame(), feature_cols
        )
    
    # ============================================================
    # TRAIN META-MODEL (TEXT ONLY - THE BASELINE)
    # ============================================================
    text_meta_model = None
    text_selected_features = None
    
    if args.test_frac == 0:
        # Cross-validation only
        results = train_meta_model_cv(
            train_features, val_features, test_features, feature_cols, args, out_dir
        )
        # Load the final text model
        model_path = out_dir / "final_cv_model.joblib"
        if model_path.exists():
            text_meta_model = joblib.load(model_path)
            selected_df = pd.read_csv(out_dir / "final_selected_features.csv")
            text_selected_features = selected_df["feature"].tolist()
            print(f"  Loaded text model from CV training: {model_path}")
    else:
        # Held-out test mode
        test_features.to_csv(out_dir / "meta_test_features.csv", index=False)
        results = train_meta_model_with_cv_selection(
            train_features, val_features, test_features, feature_cols, args, out_dir
        )
        model_path = out_dir / "meta_model.joblib"
        if model_path.exists():
            text_meta_model = joblib.load(model_path)
            selected_df = pd.read_csv(out_dir / "selected_embedding_features.csv")
            text_selected_features = selected_df["feature"].tolist()
            print(f"  Loaded text model from held-out training: {model_path}")
    
    # ============================================================
    # AUDIO FEATURE FUSION EXPERIMENTS
    # ============================================================
    if args.audio_features_csv is not None:
        print("\n" + "="*60)
        print("RUNNING AUDIO FEATURE FUSION EXPERIMENTS")
        print("="*60)
        try:
            audio_df = load_audio_features(
                args.audio_features_csv,
                speaker_col="speaker_id",
                exclude_cols=args.audio_feature_cols
            )
            
            if args.audio_feature_cols is not None:
                keep_cols = ["speaker_id"] + args.audio_feature_cols
                missing = [c for c in args.audio_feature_cols if c not in audio_df.columns]
                if missing:
                    print(f"Warning: Some specified audio feature columns not found: {missing}")
                audio_df = audio_df[keep_cols].copy()
            
            # Run ALL fusion experiments with the existing text model
            fusion_results = run_fusion_experiments(
                train_features, val_features, test_features,
                feature_cols, audio_df, args, out_dir,
                text_meta_model=text_meta_model,
                text_selected_features=text_selected_features
            )
            
            print("\nFusion experiments completed. Results saved in fusion_results/")
            
        except Exception as e:
            print(f"Error during audio fusion: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo audio features CSV provided. Skipping fusion experiments.")
    
    # Cleanup
    cleanup_temp_dirs(out_dir)
    
    # Print ensemble info
    if args.use_ensemble:
        print("\n" + "=" * 50)
        print("ENSEMBLE SUMMARY")
        print("=" * 50)
        print(f"Models in ensemble: {args.ensemble_models}")
        print(f"Voting type: {'soft (probability averaging)' if args.task == 'classification' else 'average'}")
        if args.test_frac == 0:
            print(f"Final CV model: {out_dir / 'final_cv_model.joblib'}")
        else:
            print(f"Final model: {out_dir / 'meta_model.joblib'}")


if __name__ == "__main__":
    main()