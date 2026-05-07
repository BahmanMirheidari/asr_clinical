from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int, task: str):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.task == "classification":
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def load_model(model_name: str, task: str, num_labels: int, metadata: dict):
    if task == "classification":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={int(k): v for k, v in metadata["id2label"].items()},
            label2id=metadata["label2id"],
        )
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
