# ASR Clinical Text Classifier/Regressor

Small, speaker-safe training template for ASR transcripts from Q1-Q13/Q14 tasks.

## Inputs

ASR text file:

```text
UTT_ID;TEXT
X_123_2025_01_01_07_24_Q1;the transcribed response...
X_123_2025_01_01_07_24_Q2;another response...
```

Demographics CSV:

```text
speaker_id,class1,score1,score2
X_123,MCI,12,4.5
X_124,HC,,2.1
```

`speaker_id` must match the first two underscore-separated parts of `UTT_ID`, for example `X_123`.

## Classification

```bash
python -m asr_clinical.train \
  --asr-file data/asr.txt \
  --demo-file data/demographic_info.csv \
  --target-column class1 \
  --task classification \
  --model-name distilroberta-base \
  --output-dir outputs/class1
```

## Regression

Rows with missing scores are ignored automatically.

```bash
python -m asr_clinical.train \
  --asr-file data/asr.txt \
  --demo-file data/demographic_info.csv \
  --target-column score1 \
  --task regression \
  --model-name distilroberta-base \
  --output-dir outputs/score1
```

## Useful Options

```bash
--aggregate-level speaker      # question/session/speaker metrics
--text-mode question           # question rows
--text-mode session_concat     # concatenate Q1-Q14 per session
--num-folds 5
--test-size 0.1
--loss focal                   # classification only: ce or focal
--class-weights balanced       # classification only: none or balanced
--splits-folder splits         # use existing foldN_train/val/test speaker CSVs
--filter-questions Q1 Q2 Q8    # keep only selected questions
--question-importance
```

The train/test split and CV folds are grouped by speaker to avoid speaker leakage.

## Existing Split Files

If you already generated speaker-safe splits, place files like this in one folder:

```text
splits/fold0_train.csv
splits/fold0_val.csv
splits/fold0_test.csv
splits/fold1_train.csv
splits/fold1_val.csv
splits/fold1_test.csv
```

Each split file must contain a `speaker_id` column. Other columns such as `class1`
can be present and are ignored for splitting.

```bash
python -m asr_clinical.train \
  --asr-file data/asr.txt \
  --demo-file data/demographic_info.csv \
  --target-column class1 \
  --task classification \
  --splits-folder splits \
  --output-dir outputs/class1_external_splits
```

For each fold, the model trains on `foldN_train.csv`, early-stops on
`foldN_val.csv`, and evaluates on `foldN_test.csv` if it exists. If no test file
exists for a fold, validation predictions are used as that fold's CV prediction.

After fold evaluation, a final model is also trained the same way as the default
workflow: the full usable dataset is split into a fresh speaker-safe final holdout
using `--test-size`, then `final_model` is trained on the remaining trainval data
and evaluated on that held-out final test set.

If a fold already has a saved `model/`, training for that fold is skipped. Missing
prediction or metric files are regenerated from the saved model. Existing
`final_model/model` is handled the same way.

## SHAP Question Importance

After training with question-level inputs, run SHAP over the saved prediction table:

```bash
python -m asr_clinical.shap_questions \
  --predictions-file outputs/class1/cv_predictions.csv \
  --task classification \
  --group-level speaker \
  --output-file outputs/class1/shap_question_importance.csv
```

This explains which questions contribute most to a small meta-model trained on the
per-question probabilities. The built-in `--question-importance` option in training
also writes ablation importance, which is often the cleaner sanity check.

## Inference

Create a JSON file with one dictionary per person/session:

```json
[
  {
    "Q1": "the participant answer to question one",
    "Q2": "the participant answer to question two"
  },
  {
    "Q1": "another participant answer",
    "Q3": "missing questions are fine"
  }
]
```

Run classification inference:

```bash
python -m asr_clinical.inference \
  --model-path outputs/class1 \
  --input-json data/new_answers.json \
  --output-json outputs/class1/new_predictions.json
```

You can also call it from Python:

```python
from asr_clinical.inference import predict_answers

results = predict_answers(
    answers=[{"Q1": "text", "Q2": "more text"}],
    model_path="outputs/class1",
)
```
