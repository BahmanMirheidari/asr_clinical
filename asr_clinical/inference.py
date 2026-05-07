from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def choose_device(device: str = "auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dir(path: str | Path) -> Path:
    path = Path(path)
    candidates = [
        path,
        path / "model",
        path / "final_model" / "model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a Hugging Face model config under {path}, "
        f"{path / 'model'}, or {path / 'final_model' / 'model'}."
    )


def load_labels(model_dir: Path, output_dir: Path | None = None) -> list[str]:
    metadata_paths = []
    if output_dir is not None:
        metadata_paths.append(output_dir / "metadata.json")
    metadata_paths.extend(
        [
            model_dir.parent.parent / "metadata.json",
            model_dir.parent / "metadata.json",
        ]
    )
    for metadata_path in metadata_paths:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "labels" in metadata:
                return list(metadata["labels"])

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if id2label:
        return [id2label[str(i)] for i in range(len(id2label))]
    raise ValueError("Could not infer class labels from metadata.json or model config.json.")


def normalize_answers(answer_set: dict[str, str]) -> list[tuple[str, str]]:
    rows = []
    for question_id, text in answer_set.items():
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        rows.append((str(question_id).upper(), text))
    return sorted(rows, key=lambda item: item[0])


def make_texts(answer_set: dict[str, str], text_mode: str) -> tuple[list[str], list[str]]:
    rows = normalize_answers(answer_set)
    if not rows:
        return [], []
    if text_mode == "session_concat":
        text = "\n".join(f"{question_id}: {text}" for question_id, text in rows)
        return [text], ["SESSION"]
    return [text for _, text in rows], [question_id for question_id, _ in rows]


class ASRClinicalClassifier:
    def __init__(
        self,
        model_path: str | Path,
        max_length: int = 256,
        batch_size: int = 16,
        text_mode: str | None = None,
        device: str = "auto",
    ):
        self.output_dir = Path(model_path)
        self.model_dir = resolve_model_dir(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = choose_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.labels = load_labels(self.model_dir, self.output_dir)
        self.text_mode = text_mode or self._load_text_mode(default="question")

    def _load_text_mode(self, default: str) -> str:
        for config_path in [
            self.output_dir / "config.json",
            self.model_dir.parent.parent / "config.json",
            self.model_dir.parent / "config.json",
        ]:
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                return config.get("text_mode", default)
        return default

    @torch.no_grad()
    def predict(self, answers: list[dict[str, str]]) -> list[dict[str, Any]]:
        results = []
        for item_index, answer_set in enumerate(answers):
            texts, question_ids = make_texts(answer_set, self.text_mode)
            if not texts:
                results.append(
                    {
                        "index": item_index,
                        "predicted_class": None,
                        "probabilities": {label: None for label in self.labels},
                        "question_predictions": [],
                        "error": "No non-empty answers were provided.",
                    }
                )
                continue

            probs = self._predict_texts(texts)
            mean_probs = probs.mean(axis=0)
            pred_idx = int(mean_probs.argmax())
            results.append(
                {
                    "index": item_index,
                    "predicted_class": self.labels[pred_idx],
                    "probabilities": {
                        label: float(mean_probs[i]) for i, label in enumerate(self.labels)
                    },
                    "question_predictions": [
                        {
                            "question_id": question_ids[row_idx],
                            "predicted_class": self.labels[int(row_probs.argmax())],
                            "probabilities": {
                                label: float(row_probs[label_idx])
                                for label_idx, label in enumerate(self.labels)
                            },
                        }
                        for row_idx, row_probs in enumerate(probs)
                    ],
                }
            )
        return results

    def _predict_texts(self, texts: list[str]) -> np.ndarray:
        all_probs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            logits = self.model(**enc).logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)


def predict_answers(
    answers: list[dict[str, str]],
    model_path: str | Path,
    max_length: int = 256,
    batch_size: int = 16,
    text_mode: str | None = None,
    device: str = "auto",
) -> list[dict[str, Any]]:
    classifier = ASRClinicalClassifier(
        model_path=model_path,
        max_length=max_length,
        batch_size=batch_size,
        text_mode=text_mode,
        device=device,
    )
    return classifier.predict(answers)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run classification inference on lists of answered questions."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Training output dir, final_model dir, or Hugging Face model dir.",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing a list like [{'Q1': 'text', 'Q2': 'text'}, ...].",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-mode", choices=["question", "session_concat"], default=None)
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    answers = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(answers, list):
        raise SystemExit("input-json must contain a list of question-answer dictionaries.")

    results = predict_answers(
        answers=answers,
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        text_mode=args.text_mode,
        device=args.device,
    )
    output = json.dumps(results, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
