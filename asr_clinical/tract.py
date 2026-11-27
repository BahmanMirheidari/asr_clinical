"""Two-stage CoT + regression-aware training for causal language models.

This is a practical TRACT-style path for continuous clinical scores: teacher
rationales supervise stage one; stage one's transcript-only rationales supervise
stage two. A fold-local expected-value head supplies the regression objective.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .losses import build_y_grid, decode_expected_value, raft_expected_value_loss


class ExpectedScoreHead(nn.Module):
    def __init__(self, hidden_size: int, grid_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, grid_size)

    def forward(self, hidden):
        # The causal model may run in BF16 on CUDA while the small regression
        # head is intentionally kept in FP32 for stable RAFT probabilities.
        # Casting the activation preserves gradients through the cast.
        return self.projection(hidden.float())


def _hidden_size(config):
    value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(config, "n_embd", None)
    if value is None:
        raise ValueError("Causal-LM config does not expose hidden_size or n_embd")
    return int(value)


def _load_kwargs(device, local_files_only=False):
    kwargs = {"local_files_only": local_files_only}
    if torch.device(device).type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def prompt_for(text: str) -> str:
    return ("Assess the supplied speech transcript for language-related evidence "
            "that may be relevant to cognitive screening. Do not diagnose or "
            "infer unsupported facts. Explain uncertainty briefly.\n\n"
            f"Transcript:\n{text}\n\nRationale:")


def _prompt_ids(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        return ids.tolist() if hasattr(ids, "tolist") else ids
    return tokenizer(prompt, add_special_tokens=True)["input_ids"]


def _answer_ids(tokenizer, prompt, answer):
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False)
        prefix = tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True)
        full = full.tolist() if hasattr(full, "tolist") else full
        prefix = prefix.tolist() if hasattr(prefix, "tolist") else prefix
        return full, len(prefix)
    full = tokenizer(prompt + " " + answer, add_special_tokens=True)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    return full, len(prefix)


def _pad_id_rows(tokenizer, rows, max_length):
    clipped = []
    for ids in rows:
        if len(ids) > max_length:
            ids = ids[:max_length]
        clipped.append({"input_ids": ids})
    return tokenizer.pad(clipped, padding=True, max_length=max_length,
                         return_tensors="pt")


def _last_hidden(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
    positions = attention_mask.sum(dim=1).clamp(min=1) - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    return outputs.hidden_states[-1][rows, positions], outputs


def _attach_head(model, grid_size, device, head_state=None):
    hidden_size = _hidden_size(model.config)
    model.tract_score_head = ExpectedScoreHead(hidden_size, grid_size)
    if head_state is not None:
        model.tract_score_head.load_state_dict(head_state)
    model.to(device)
    return model


def load_tract_model(model_dir: Path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, **_load_kwargs(device, local_files_only=True)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg_path = model_dir / "tract_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing TRACT config: {cfg_path}")
    info = json.loads(cfg_path.read_text(encoding="utf-8"))
    state = torch.load(model_dir / "tract_score_head.pt", map_location=device)
    _attach_head(model, len(info["y_grid"]), device, state)
    model.tract_y_grid = info["y_grid"]
    model.tract_temperature = float(info.get("temperature", 1.0))
    model.eval()
    return model, tokenizer


def _generate_rationales(model, tokenizer, texts, cfg, device):
    model.eval()
    results = []
    for text in texts:
        prompt = prompt_for(str(text))
        prompt_ids = _prompt_ids(tokenizer, prompt)[-cfg.max_length:]
        encoded = tokenizer.pad([{"input_ids": prompt_ids}],
                                return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=cfg.tract_generation_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        results.append(tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip())
    return results


def _train_stage(model, tokenizer, head, texts, rationales, targets, grid,
                 cfg, device, seed):
    if len(texts) != len(rationales) or len(texts) != len(targets):
        raise ValueError("TRACT texts, rationales, and targets must align")
    if any(not str(r).strip() for r in rationales):
        raise ValueError("TRACT rationale column contains empty values")
    model.train()
    head.train()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)
    optimizer.add_param_group({"params": head.parameters()})
    rng = random.Random(seed)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device)
    max_len = max(cfg.max_length, 256) + cfg.tract_generation_tokens

    for epoch in range(cfg.epochs):
        order = list(range(len(texts)))
        rng.shuffle(order)
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            prompts = [prompt_for(str(texts[i])) for i in idx]
            pairs = [_answer_ids(tokenizer, prompts[j], str(rationales[i]))
                     for j, i in enumerate(idx)]
            input_rows = [pair[0] for pair in pairs]
            prefix_lengths = [pair[1] for pair in pairs]
            encoded = _pad_id_rows(tokenizer, input_rows, max_len)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = encoded["input_ids"].clone()
            for j, prompt_len in enumerate(prefix_lengths):
                labels[j, :prompt_len] = -100
            labels[encoded["attention_mask"] == 0] = -100

            optimizer.zero_grad(set_to_none=True)
            hidden, outputs = _last_hidden(
                model, encoded["input_ids"], encoded["attention_mask"])
            ce_loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1].contiguous().reshape(
                    -1, outputs.logits.shape[-1]),
                labels[:, 1:].contiguous().reshape(-1), ignore_index=-100)
            score_logits = head(hidden)
            raft_loss = raft_expected_value_loss(
                score_logits, target_tensor[idx], grid_tensor,
                cfg.raft_temperature)
            loss = ce_loss + cfg.raft_weight * raft_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                cfg.gradient_clip_val)
            optimizer.step()
        print(f"TRACT stage epoch {epoch + 1}/{cfg.epochs}: "
              f"CE={float(ce_loss.detach()):.4f} "
              f"RAFT={float(raft_loss.detach()):.4f}")
    model.eval()
    head.eval()


def fit_two_stage_tract(train_df, cfg, fold_dir: Path, device):
    if "rationale" not in train_df:
        raise ValueError("TRACT training requires reviewed 'rationale' values")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["label"].astype(float).to_numpy()
    teacher_rationales = train_df["rationale"].astype(str).tolist()
    grid = build_y_grid(targets, cfg.y_grid_size,
                        cfg.y_grid_min, cfg.y_grid_max).tolist()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Stage 1: learn teacher-provided rationales and the fold-local score grid.
    stage1 = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, **_load_kwargs(device)).to(device)
    head1 = ExpectedScoreHead(_hidden_size(stage1.config), len(grid)).to(device)
    _train_stage(stage1, tokenizer, head1, texts, teacher_rationales,
                 targets, grid, cfg, device, cfg.seed)
    self_rationales = _generate_rationales(stage1, tokenizer, texts, cfg, device)
    empty_generated = [i for i, rationale in enumerate(self_rationales)
                       if not str(rationale).strip()]
    if empty_generated:
        print(
            f"[TRACT] Stage-one generation returned empty rationales for "
            f"{len(empty_generated)}/{len(self_rationales)} training rows; "
            "using their reviewed teacher rationales for stage two."
        )
        for i in empty_generated:
            self_rationales[i] = teacher_rationales[i]
    del stage1, head1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: reinitialize from the seed model and train on stage-one CoTs.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, **_load_kwargs(device)).to(device)
    head = ExpectedScoreHead(_hidden_size(model.config), len(grid)).to(device)
    _train_stage(model, tokenizer, head, texts, self_rationales,
                 targets, grid, cfg, device, cfg.seed + 1)
    model.tract_score_head = head
    model.tract_y_grid = grid
    model.tract_temperature = cfg.raft_temperature
    return model, tokenizer, grid


@torch.no_grad()
def predict_tract(model, tokenizer, texts, batch_size, max_length, device,
                  y_grid=None, temperature=1.0, generation_tokens=128):
    rationales = []
    values = []
    grid = torch.as_tensor(y_grid or model.tract_y_grid, dtype=torch.float32,
                           device=device)
    model.eval()
    head = model.tract_score_head.to(device).eval()
    for start in range(0, len(texts), max(1, batch_size)):
        chunk = texts[start:start + max(1, batch_size)]
        generated_rationales = []
        for text in chunk:
            prompt = prompt_for(str(text))
            prompt_ids = _prompt_ids(tokenizer, prompt)[-max_length:]
            enc = tokenizer.pad([{"input_ids": prompt_ids}], return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(
                **enc, max_new_tokens=generation_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id)
            rationale = tokenizer.decode(
                out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            generated_rationales.append(rationale)
            rationales.append(rationale)
        context_rows = [_answer_ids(tokenizer, prompt_for(str(text)), rationale)[0]
                        for text, rationale in zip(chunk, generated_rationales)]
        enc = _pad_id_rows(tokenizer, context_rows, max_length + generation_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden, _ = _last_hidden(model, enc["input_ids"], enc["attention_mask"])
        logits = head(hidden)
        values.extend(decode_expected_value(logits, grid, temperature).cpu().tolist())
    return np.asarray(values, dtype=float), rationales
