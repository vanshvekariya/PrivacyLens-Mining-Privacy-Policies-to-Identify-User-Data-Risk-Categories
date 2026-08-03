import json
import os
import platform
import random
import sys
import time
from pathlib import Path

# Keep HuggingFace downloads inside the project
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from privacylens.config import (CATEGORIES, MAX_SEQ_LEN, POS_WEIGHT_CAP, SEED,
                                TRANSFORMER_CHECKPOINT,
                                TRANSFORMER_FOLDS_TO_RUN,
                                TRANSFORMER_METADATA_JSON,
                                TRANSFORMER_MODEL_NAME, TRANSFORMER_OOF_CSV,
                                TRANSFORMER_PARAMS)
from privacylens.prediction import folds_to_indices, load_or_build_folds

# Data file paths
ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_CSV = ROOT / "data" / "segments.csv"
RESULTS = ROOT / "results"
FOLD_RUN_DIR = RESULTS / "transformer_runs"
REPORT_FILE = RESULTS / "transformer_results.txt"

# Set seeds
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def pick_device():
    # Pick the best available device: cuda, mps, or cpu
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


device = pick_device()


def sanitize_col(category):
    # Convert a category name into a csv-safe probability column
    return "prob_" + (category.replace("&", "and").replace("/", "_")
                              .replace("-", "_").replace(" ", "_"))


PROB_COLS = [sanitize_col(c) for c in CATEGORIES]


class SegmentDataset(Dataset):
    # Wraps texts and multi-hot labels into a torch dataset
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(list(texts), truncation=True,
                                   padding=False, max_length=max_len)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.as_tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item


class MultiLabelTrainer(Trainer):
    # Trainer with BCEWithLogitsLoss and per-fold pos_weight
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = nn.BCEWithLogitsLoss(pos_weight=w)(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def load_train_data():
    # Load the train split from segments.csv and build the multi-hot label matrix
    df = pd.read_csv(SEGMENTS_CSV)
    train = df[df["split"] == "train"].reset_index(drop=True)
    texts = train["text"].fillna("").tolist()
    label_sets = [s.split("|") for s in train["labels"].fillna("")]
    groups = train["policy_stem"].astype(str).values
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y = mlb.fit_transform(label_sets).astype(np.int64)
    return texts, Y, groups, train


def compute_pos_weight(Y_train_fold):
    # pos_weight per category = neg / pos, capped at POS_WEIGHT_CAP
    pos = Y_train_fold.sum(axis=0).astype(np.float64)
    neg = Y_train_fold.shape[0] - pos
    pos = np.where(pos == 0, 1.0, pos)
    raw = neg / pos
    return torch.tensor(np.minimum(raw, POS_WEIGHT_CAP), dtype=torch.float32)


def compute_metrics(eval_pred):
    # Multi-label macro / micro f1 at threshold 0.5
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
    }


def audit_token_lengths(texts, tokenizer, sample_n=500):
    # Sample-based tokenizer length distribution
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(texts), size=min(sample_n, len(texts)), replace=False)
    lens = np.array([len(tokenizer.encode(texts[i], add_special_tokens=True,
                                          truncation=False)) for i in idx])
    return {
        "sample_n": int(len(lens)),
        "p50": float(np.percentile(lens, 50)),
        "p95": float(np.percentile(lens, 95)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
        "truncated_frac": float((lens > MAX_SEQ_LEN).mean()),
    }


def preload_prior_folds(fold_ids, train_df):
    # Load predictions and metadata for folds not being re-run, so a partial
    # run merges with earlier ones instead of overwriting them
    n_rows = len(train_df)
    oof_probs = np.full((n_rows, len(CATEGORIES)), np.nan, dtype=np.float32)
    fold_of_row = np.full(n_rows, -1, dtype=np.int64)
    per_fold_meta = []

    if TRANSFORMER_OOF_CSV.exists():
        prior = pd.read_csv(TRANSFORMER_OOF_CSV)
        prior = prior[~prior["fold_id"].isin(fold_ids)]
        if len(prior):
            key_to_idx = {(s, int(i)): idx for idx, (s, i) in
                          enumerate(zip(train_df["policy_stem"], train_df["segment_id"]))}
            for _, row in prior.iterrows():
                key = (row["policy_stem"], int(row["segment_id"]))
                if key not in key_to_idx:
                    continue
                idx = key_to_idx[key]
                fold_of_row[idx] = int(row["fold_id"])
                for j, col in enumerate(PROB_COLS):
                    oof_probs[idx, j] = row[col]

    if TRANSFORMER_METADATA_JSON.exists():
        try:
            with open(TRANSFORMER_METADATA_JSON) as f:
                prior_meta = json.load(f)
            per_fold_meta = [pf for pf in prior_meta.get("per_fold", [])
                             if pf.get("fold_id") not in fold_ids]
        except Exception:
            per_fold_meta = []

    return oof_probs, fold_of_row, per_fold_meta


def train_one_fold(fold_id, tr_idx, va_idx, texts, Y, tokenizer):
    # Fine-tune the transformer on one fold and return validation probabilities
    set_seed(SEED)
    fold_dir = FOLD_RUN_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SegmentDataset([texts[i] for i in tr_idx], Y[tr_idx],
                              tokenizer, MAX_SEQ_LEN)
    val_ds = SegmentDataset([texts[i] for i in va_idx], Y[va_idx],
                            tokenizer, MAX_SEQ_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_CHECKPOINT,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(CATEGORIES)},
        label2id={c: i for i, c in enumerate(CATEGORIES)},
    )

    pos_weight = compute_pos_weight(Y[tr_idx])

    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        **TRANSFORMER_PARAMS,
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Per-epoch validation history
    history = [log for log in trainer.state.log_history if "eval_macro_f1" in log]

    # Out-of-fold predictions on the validation fold
    pred_out = trainer.predict(val_ds)
    probs = 1.0 / (1.0 + np.exp(-pred_out.predictions))

    del trainer, model
    if device == "mps":
        torch.mps.empty_cache()

    return probs, history, total_time, pos_weight.tolist()


def write_oof_predictions(oof_probs, fold_of_row, train_df):
    # Write out-of-fold predictions and per-category probabilities to csv
    kept = fold_of_row >= 0
    n_kept = int(kept.sum())
    out = train_df.loc[kept, ["policy_stem", "segment_id", "labels"]].copy()
    out.rename(columns={"labels": "gold_labels"}, inplace=True)
    out["fold_id"] = fold_of_row[kept]
    kept_probs = oof_probs[kept]
    for j, col in enumerate(PROB_COLS):
        out[col] = kept_probs[:, j]
    preds = (kept_probs >= 0.5).astype(int)
    out["predicted_labels_at_0_5"] = [
        "|".join(c for j, c in enumerate(CATEGORIES) if preds[i, j])
        for i in range(len(preds))
    ]
    TRANSFORMER_OOF_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRANSFORMER_OOF_CSV, index=False)
    return n_kept


def write_metadata(fold_ids, per_fold_meta, length_audit, n_train, n_kept):
    # Write metadata json with versions, params, and per-fold history
    metadata = {
        "model_name": TRANSFORMER_MODEL_NAME,
        "checkpoint": TRANSFORMER_CHECKPOINT,
        "max_seq_len": MAX_SEQ_LEN,
        "pos_weight_cap": POS_WEIGHT_CAP,
        "training_params": TRANSFORMER_PARAMS,
        "seed": SEED,
        "n_folds_run": len(fold_ids),
        "fold_ids_run": list(fold_ids),
        "n_train_segments_total": n_train,
        "n_train_segments_predicted": n_kept,
        "categories": list(CATEGORIES),
        "token_length_audit": length_audit,
        "device": device,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "per_fold": per_fold_meta,
    }
    with open(TRANSFORMER_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_report_to_file(fold_ids, per_fold_meta, length_audit):
    # Write results into a text file
    with open(REPORT_FILE, 'w') as f:
        f.write('### DISTILBERT FINE-TUNING ###\n\n')
        f.write(f'checkpoint: {TRANSFORMER_CHECKPOINT}\n')
        f.write(f'max seq len: {MAX_SEQ_LEN}\n')
        f.write(f'device: {device}\n')
        f.write(f'seed: {SEED}\n')
        f.write(f'folds run: {list(fold_ids)}\n\n')

        f.write('### TOKEN LENGTH AUDIT ###\n\n')
        for k, v in length_audit.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')

        for fm in per_fold_meta:
            f.write(f'### FOLD {fm["fold_id"]} ###\n\n')
            f.write(f'train segments: {fm["n_train"]}\n')
            f.write(f'val segments: {fm["n_val"]}\n')
            f.write(f'training time: {fm["seconds"]:.1f} seconds\n')
            for h in fm['history']:
                f.write(f'  epoch {h.get("epoch")}: '
                        f'macro_f1={h.get("eval_macro_f1"):.4f} '
                        f'micro_f1={h.get("eval_micro_f1"):.4f}\n')
            f.write('\n')


def main():
    # Load train data
    print('Loading train data')
    texts, Y, groups, train_df = load_train_data()
    print(f'Loaded {len(texts)} train segments across {len(set(groups))} policies')

    # Load tokenizer
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_CHECKPOINT)

    # Audit token lengths
    print('Auditing token lengths')
    length_audit = audit_token_lengths(texts, tokenizer)
    print(f'Token lengths -- p50 {length_audit["p50"]:.0f}, '
          f'p95 {length_audit["p95"]:.0f}, p99 {length_audit["p99"]:.0f}, '
          f'truncated {length_audit["truncated_frac"]:.1%}')

    # Load policy-grouped cross-validation folds
    print('Loading fold assignments')
    fold_of_policy = load_or_build_folds(groups)
    fold_iter = list(folds_to_indices(groups, fold_of_policy))
    all_fold_ids = list(range(len(fold_iter)))
    fold_ids = TRANSFORMER_FOLDS_TO_RUN if TRANSFORMER_FOLDS_TO_RUN is not None else all_fold_ids

    # Fine-tune each fold and accumulate out-of-fold predictions
    oof_probs, fold_of_row, per_fold_meta = preload_prior_folds(fold_ids, train_df)

    for f in fold_ids:
        tr_idx, va_idx = fold_iter[f]
        print(f'\nTraining fold {f}: {len(tr_idx)} train / {len(va_idx)} val')
        probs, history, total_time, pos_weight = train_one_fold(
            f, tr_idx, va_idx, texts, Y, tokenizer)
        oof_probs[va_idx] = probs
        fold_of_row[va_idx] = f
        per_fold_meta.append({
            'fold_id': f,
            'n_train': int(len(tr_idx)),
            'n_val': int(len(va_idx)),
            'seconds': total_time,
            'history': history,
            'pos_weight': pos_weight,
        })
        last = history[-1] if history else {}
        print(f'Fold {f} done in {total_time:.1f}s '
              f'(macro_f1={last.get("eval_macro_f1", float("nan")):.4f})')

    # Write out-of-fold predictions
    print('\nWriting out-of-fold predictions')
    n_kept = write_oof_predictions(oof_probs, fold_of_row, train_df)
    print(f'Wrote {n_kept} rows to {TRANSFORMER_OOF_CSV.name}')

    # Write metadata json
    print('Writing metadata')
    write_metadata(fold_ids, per_fold_meta, length_audit, len(texts), n_kept)
    print(f'Wrote {TRANSFORMER_METADATA_JSON.name}')

    # Write text report
    print('Writing text report')
    write_report_to_file(fold_ids, per_fold_meta, length_audit)
    print(f'Wrote {REPORT_FILE.name}')


# Call the main function
main()
