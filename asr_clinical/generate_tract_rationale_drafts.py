"""Generate local, reviewable ECAS rationale drafts for TRACT training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import read_asr_file


ECAS_CONTEXT_COLUMNS = [
    "ecas_fluency", "ecas_exec", "ecas_lang", "ecas_mem", "ecas_vis",
    "ecas_als_total", "ecas_nonals_total",
]
DEFAULT_RUBRIC = """ECAS total is out of 136; higher scores indicate better performance.
Under UK English-version guidance, a total score at or below 105 suggests
possible cognitive impairment on screening; a score above 105 is above that
cutoff and does not prove normal cognition. Domain cutoffs for possible
abnormality: fluency 14/24, executive 33/48, language 26/28, memory 13/24,
visuospatial 10/12, ALS-specific total 77/100, and non-ALS-specific total 24/36.
These are screening thresholds, not diagnoses. Do not infer a diagnosis, cause,
or ability from a transcript. Explicitly distinguish score findings from
transcript observations and say when the transcript cannot explain the score.
"""


def _prompt(row, score_column, context_columns, rubric):
    scores = "\n".join(f"{col}: {row[col]}" for col in context_columns)
    return f"""Write a brief, cautious research annotation (2-4 sentences) for this
ECAS assessment profile. Use only the supplied scores and transcript. Do not
claim the transcript caused or proves the assessment result. State which domain
scores meet the supplied cutoffs, if any. Do not diagnose. If the transcript
does not explain the score profile, say so. Return only the rationale text.

Scoring rubric:
{rubric.strip()}

Target score ({score_column}): {row[score_column]}
ECAS domain scores:
{scores}

Transcript:
{row['text']}"""


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate ECAS rationale drafts with a local HF causal model."
    )
    p.add_argument("--asr-file", required=True)
    p.add_argument("--demo-file", required=True)
    p.add_argument("--target-column", default="ecas_total")
    p.add_argument("--context-columns", nargs="+", default=ECAS_CONTEXT_COLUMNS,
                   help="ECAS fields from demo CSV supplied alongside the target")
    p.add_argument("--score-rubric-file", default=None,
                   help="Optional editable plain-text score/range rubric")
    p.add_argument("--teacher-model", required=True,
                   help="Hugging Face model ID or local causal instruct model directory")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--delimiter", default=";")
    p.add_argument("--text-index", type=int, default=-1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--include-prompt", action="store_true")
    return p


def _device(name):
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def main():
    args = build_parser().parse_args()
    asr = read_asr_file(args.asr_file, delimiter=args.delimiter,
                        text_index=args.text_index)
    demo = pd.read_csv(args.demo_file)
    if "speaker_id" not in demo.columns:
        raise ValueError("demo CSV must contain speaker_id")
    required = [args.target_column, *args.context_columns]
    missing = [name for name in required if name not in demo.columns]
    if missing:
        raise ValueError(f"demo CSV is missing requested score columns: {missing}")
    demo["speaker_id"] = demo["speaker_id"].astype(str).str.strip()
    asr["speaker_id"] = asr["speaker_id"].astype(str).str.strip()
    merged = asr.merge(demo[["speaker_id", *required]], on="speaker_id", how="inner")
    merged = merged[merged[args.target_column].notna() & merged["text"].notna()].copy()
    if merged.empty:
        raise ValueError("No ASR rows matched a demo row with an ECAS total")
    if merged["utterance_id"].duplicated().any():
        raise ValueError("ASR utterance_id values must be unique for review/import")

    rubric = (Path(args.score_rubric_file).read_text(encoding="utf-8")
              if args.score_rubric_file else DEFAULT_RUBRIC)
    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    load_kwargs = {}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, **load_kwargs
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts, drafts = [], []
    for i, (_, row) in enumerate(merged.iterrows(), start=1):
        prompt = _prompt(row, args.target_column, args.context_columns, rubric)
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt")
            input_ids = input_ids[:, :args.max_input_tokens]
            encoded = {"input_ids": input_ids,
                       "attention_mask": torch.ones_like(input_ids)}
        else:
            encoded = tokenizer(prompt, truncation=True,
                                max_length=args.max_input_tokens,
                                return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        drafts.append(tokenizer.decode(
            generated[0, input_len:], skip_special_tokens=True).strip())
        prompts.append(prompt)
        if i % 25 == 0 or i == len(merged):
            print(f"Generated {i}/{len(merged)} drafts")

    merged["rationale_draft"] = drafts
    merged["rationale_final"] = ""
    merged["rationale_review_status"] = "needs_review"
    if args.include_prompt:
        merged["rationale_prompt"] = prompts
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Review drafts in {output}; put approved text in rationale_final.")


if __name__ == "__main__":
    main()
