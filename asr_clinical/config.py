from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    asr_file: str
    demo_file: str
    target_column: str
    task: str
    output_dir: str
    model_name: str = "distilroberta-base"
    text_mode: str = "question"
    aggregate_level: str = "speaker"
    num_folds: int = 5
    test_size: float = 0.1
    final_dev_size: float = 0.1
    seed: int = 42
    max_length: int = 256
    batch_size: int = 8
    eval_batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    patience: int = 2
    class_weights: str = "balanced"
    loss: str = "ce"
    focal_gamma: float = 2.0
    folds_file: str | None = None
    splits_folder: str | None = None
    filter_questions: list[str] | None = None
    question_importance: bool = False
    min_text_chars: int = 1
    regression_metric: str = "rmse"
    label_smoothing: float = 0.0
    gradient_clip_val: float = 1.0
    dropout_rate: float = 0.1
    y_mean: float = 0.0
    y_std:  float = 1.0
    normalize_regression_targets: bool = False
    force_retrain: bool = False
    text_index: int = -1
    use_raft: bool = False
    y_grid_size: int = 101
    y_grid_min: float | None = None
    y_grid_max: float | None = None
    raft_temperature: float = 1.0
    raft_weight: float = 1.0
    raft_y_grid: list[float] | None = None
    use_tract: bool = False
    rationale_file: str | None = None
    rationale_column: str = "rationale_final"
    tract_generation_tokens: int = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train speaker-safe classification/regression models on ASR transcripts."
    )
    parser.add_argument("--asr-file", required=True)
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--text-mode", choices=["question", "session_concat"], default="question")
    parser.add_argument(
        "--aggregate-level",
        choices=["question", "session", "speaker"],
        default="speaker",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--final-dev-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--folds-file", default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1) 
    parser.add_argument("--patience", type=int, default=2) 


    parser.add_argument("--regression-metric", choices=["mae", "rmse"], default="rmse",
                        help="Metric to optimize for regression tasks (default: rmse)")
    parser.add_argument(
        "--splits-folder",
        default=None,
        help=(
            "Folder containing foldN_train.csv, foldN_val.csv, and optionally "
            "foldN_test.csv files with speaker_id columns."
        ),
    )
    parser.add_argument(
        "--filter-questions",
        nargs="+",
        default=None,
        help="Optional question IDs to keep, for example: --filter-questions Q1 Q2 Q8",
    )
    parser.add_argument("--question-importance", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--use-raft", action="store_true",
                        help="Use regression-aware expected-value loss on a target grid")
    parser.add_argument("--y-grid-size", type=int, default=101)
    parser.add_argument("--y-grid-min", type=float, default=None)
    parser.add_argument("--y-grid-max", type=float, default=None)
    parser.add_argument("--raft-temperature", type=float, default=1.0)
    parser.add_argument("--raft-weight", type=float, default=1.0)
    parser.add_argument("--use-tract", action="store_true",
                        help="Use two-stage causal-LM CoT + RAFT regression")
    parser.add_argument("--rationale-file", default=None,
                        help="Reviewed rationale CSV keyed by utterance_id")
    parser.add_argument("--rationale-column", default="rationale_final")
    parser.add_argument("--tract-generation-tokens", type=int, default=128)
    return parser


def parse_args() -> TrainConfig:
    args = build_parser().parse_args()
    return TrainConfig(**vars(args))
