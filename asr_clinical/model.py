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


def load_tokenizer(model_name):
    """Load tokenizer with multiple fallback strategies.""" 
    
    # Strategy 1: Try with default settings
    try:
        # For DeBERTa-v3, try fast tokenizer first with specific settings
        if "deberta" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True
            )
            # Test if it works
            test = tokenizer("test", return_tensors="pt")
            return tokenizer
        else:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer failed: {e}")
        
        # Strategy 2: Try slow tokenizer
        try:
            print(f"Attempting to load slow tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            return tokenizer
        except Exception as e2:
            print(f"Slow tokenizer also failed: {e2}")
            
            # Strategy 3: Fallback to a known working model
            print(f"Falling back to distilbert-base-uncased tokenizer")
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            return tokenizer


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
