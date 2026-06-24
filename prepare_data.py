"""Data loading and tokenization for GenomicBenchmarks fine-tuning.

Provides ``load_task``, which returns tokenized train/test splits ready for a
PyTorch DataLoader. It first tries to load the task through the Hugging Face
``datasets`` hub, then through the ``genomic_benchmarks`` PyPI package, and
finally falls back to local CSV files at ``data/{task_name}/train.csv`` and
``data/{task_name}/test.csv`` with columns ``sequence,label``.

Also provides reverse complement (RC) augmentation via ``--augment_rc``:
appends the RC of every training sequence (same label), doubling effective
training data — standard practice for double-stranded DNA tasks.

Run a data preview (no model required):

    python prepare_data.py --task human_enhancers_cohn --max_length 500
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple


# Candidate Hugging Face dataset ids to try, in order. Different mirrors of the
# GenomicBenchmarks collection expose slightly different ids/configs.
_HF_DATASET_CANDIDATES = [
    ("katarinagresova/Genomic_Benchmarks_{task}", None),
    ("genomic_benchmarks/{task}", None),
]


def _normalize_records(sequences: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """Coerce sequences to upper-case strings and labels to ints."""
    seqs = [str(s).strip().upper() for s in sequences]
    ints = [int(l) for l in labels]
    return seqs, ints


_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_RC_TABLE)[::-1]


def _augment_reverse_complement(
    sequences: List[str], labels: List[int]
) -> Tuple[List[str], List[int]]:
    """Double the dataset by appending the reverse complement of each sequence.

    The reverse complement is biologically equivalent for double-stranded DNA,
    so it carries the same label. This is standard practice for genomic models.
    """
    rc_seqs = [_reverse_complement(s) for s in sequences]
    return sequences + rc_seqs, labels + labels


def _load_from_hf(task_name: str) -> Optional[Dict[str, Tuple[List[str], List[int]]]]:
    """Try loading the task from the Hugging Face `datasets` hub.

    Returns a dict with 'train'/'test' -> (sequences, labels), or None if no
    candidate id could be loaded.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return None

    for repo, config_tmpl in _HF_DATASET_CANDIDATES:
        repo_id = repo.format(task=task_name)
        config = config_tmpl.format(task=task_name) if config_tmpl else None
        try:
            if config is not None:
                ds = load_dataset(repo_id, config)
            else:
                ds = load_dataset(repo_id)
        except Exception:
            continue

        train_split = ds.get("train") if hasattr(ds, "get") else ds["train"]
        test_split = None
        for key in ("test", "validation", "valid"):
            if key in ds:
                test_split = ds[key]
                break
        if train_split is None or test_split is None:
            continue

        seq_col = _guess_column(train_split.column_names, ("sequence", "seq", "dna", "text"))
        label_col = _guess_column(train_split.column_names, ("label", "labels", "target", "class"))
        if seq_col is None or label_col is None:
            continue

        train = _normalize_records(train_split[seq_col], train_split[label_col])
        test = _normalize_records(test_split[seq_col], test_split[label_col])
        return {"train": train, "test": test}

    return None


def _guess_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _load_from_genomic_benchmarks(task_name: str) -> Optional[Dict[str, Tuple[List[str], List[int]]]]:
    """Try loading via the `genomic_benchmarks` PyPI package."""
    try:
        from genomic_benchmarks.loc2seq import download_dataset
        from genomic_benchmarks.data_check import is_downloaded
    except Exception:
        return None

    try:
        if not is_downloaded(task_name):
            download_dataset(task_name)
        base = os.path.join(
            os.path.expanduser("~"), ".genomic_benchmarks", task_name
        )
        if not os.path.isdir(base):
            return None
        return _read_genomic_benchmarks_dirs(base)
    except Exception:
        return None


def _read_genomic_benchmarks_dirs(base: str) -> Optional[Dict[str, Tuple[List[str], List[int]]]]:
    """Read the on-disk genomic_benchmarks layout: {split}/{class}/*.txt."""
    splits: Dict[str, Tuple[List[str], List[int]]] = {}
    for split in ("train", "test"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            return None
        class_names = sorted(
            d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
        )
        label_map = {name: idx for idx, name in enumerate(class_names)}
        seqs: List[str] = []
        labels: List[int] = []
        for name in class_names:
            class_dir = os.path.join(split_dir, name)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                with open(fpath, "r", encoding="utf-8") as fh:
                    seqs.append(fh.read().strip().upper())
                    labels.append(label_map[name])
        splits[split] = (seqs, labels)
    return splits


def _load_from_csv(task_name: str) -> Dict[str, Tuple[List[str], List[int]]]:
    """Fall back to local CSV files with `sequence,label` columns."""
    base = os.path.join("data", task_name)
    out: Dict[str, Tuple[List[str], List[int]]] = {}
    for split in ("train", "test"):
        path = os.path.join(base, f"{split}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Could not load task '{task_name}' from Hugging Face, the "
                f"genomic_benchmarks package, or local CSV. Expected file: {path} "
                f"with columns 'sequence,label'."
            )
        seqs: List[str] = []
        labels: List[int] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            seq_col = _guess_column(reader.fieldnames or [], ("sequence", "seq", "dna", "text"))
            label_col = _guess_column(reader.fieldnames or [], ("label", "labels", "target", "class"))
            if seq_col is None or label_col is None:
                raise ValueError(
                    f"{path} must contain 'sequence' and 'label' columns; "
                    f"found {reader.fieldnames}."
                )
            for row in reader:
                seqs.append(row[seq_col])
                labels.append(row[label_col])
        out[split] = _normalize_records(seqs, labels)
    return out


def _load_raw(task_name: str) -> Dict[str, Tuple[List[str], List[int]]]:
    """Load raw (sequence, label) records for the task from the best source."""
    for loader in (_load_from_hf, _load_from_genomic_benchmarks):
        data = loader(task_name)
        if data is not None:
            return data
    return _load_from_csv(task_name)


def _tokenize(
    sequences: List[str],
    labels: List[int],
    max_length: int,
    tokenizer,
) -> Dict[str, "object"]:
    """Tokenize sequences into tensors using the NT 6-mer tokenizer."""
    import torch

    encoded = tokenizer(
        sequences,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    batch: Dict[str, object] = {"input_ids": encoded["input_ids"]}
    if "attention_mask" in encoded:
        batch["attention_mask"] = encoded["attention_mask"]
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch


def load_task(
    task_name: str,
    max_length: int,
    tokenizer,
    augment_rc: bool = False,
    val_frac: float = 0.0,
    seed: int = 42,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], int]:
    """Load and tokenize a GenomicBenchmarks task.

    Args:
        task_name: GenomicBenchmarks task, e.g. ``human_enhancers_cohn``.
        max_length: Max sequence length for tokenization/truncation.
        tokenizer: A loaded tokenizer (required for tokenization).
        augment_rc: If True, double training data with reverse complement sequences.
        val_frac: Fraction of training data to hold out as a validation set
            (0.0 = no validation split). The validation split is used for
            checkpoint selection so the test set is never touched during training.
        seed: Seed for the train/val split shuffle.

    Returns:
        ``(train_batch, val_batch, test_batch, num_labels)``. ``val_batch`` is
        empty (zero-length tensors) when ``val_frac`` is 0.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required to tokenize the dataset.")

    raw = _load_raw(task_name)
    train_seqs, train_labels = raw["train"]
    test_seqs, test_labels = raw["test"]

    num_labels = len(set(train_labels) | set(test_labels))

    if augment_rc:
        orig_n = len(train_seqs)
        train_seqs, train_labels = _augment_reverse_complement(train_seqs, train_labels)
        print(f"  [RC aug] train: {orig_n} → {len(train_seqs)} examples (+ reverse complements)")

    # Carve a validation split out of the training data BEFORE tokenization.
    # The test set is never used for checkpoint selection (no leakage).
    val_seqs: List[str] = []
    val_labels: List[int] = []
    if val_frac and val_frac > 0.0:
        import random
        idx = list(range(len(train_seqs)))
        random.Random(seed).shuffle(idx)
        n_val = int(len(idx) * val_frac)
        val_idx = set(idx[:n_val])
        new_train_seqs, new_train_labels = [], []
        for i, (s, l) in enumerate(zip(train_seqs, train_labels)):
            if i in val_idx:
                val_seqs.append(s)
                val_labels.append(l)
            else:
                new_train_seqs.append(s)
                new_train_labels.append(l)
        train_seqs, train_labels = new_train_seqs, new_train_labels
        print(f"  [val split] held out {len(val_seqs)} val examples "
              f"({val_frac:.0%}); train now {len(train_seqs)}")

    train_batch = _tokenize(train_seqs, train_labels, max_length, tokenizer)
    test_batch = _tokenize(test_seqs, test_labels, max_length, tokenizer)
    if val_seqs:
        val_batch = _tokenize(val_seqs, val_labels, max_length, tokenizer)
    else:
        # Empty placeholder with matching keys
        import torch as _t
        val_batch = {"input_ids": _t.empty((0, max_length), dtype=_t.long),
                     "labels": _t.empty((0,), dtype=_t.long)}
        if "attention_mask" in train_batch:
            val_batch["attention_mask"] = _t.empty((0, max_length), dtype=_t.long)

    # Shuffle test split so any eval_limit subset is class-balanced
    import torch as _t
    g = _t.Generator().manual_seed(42)
    perm = _t.randperm(test_batch["labels"].numel(), generator=g)
    for k in list(test_batch.keys()):
        test_batch[k] = test_batch[k][perm]

    return train_batch, val_batch, test_batch, num_labels


def _class_balance(labels: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a GenomicBenchmarks task (no model required)."
    )
    parser.add_argument("--task", default="human_enhancers_cohn")
    parser.add_argument("--max_length", type=int, default=500)
    args = parser.parse_args()

    raw = _load_raw(args.task)
    train_seqs, train_labels = raw["train"]
    test_seqs, test_labels = raw["test"]

    print(f"Task: {args.task}")
    print(f"  train split: {len(train_seqs)} examples")
    print(f"  test  split: {len(test_seqs)} examples")
    print(f"  num labels:  {len(set(train_labels) | set(test_labels))}")
    print(f"  train class balance: {_class_balance(train_labels)}")
    print(f"  test  class balance: {_class_balance(test_labels)}")

    print("\nExamples (train):")
    for seq, label in zip(train_seqs[:2], train_labels[:2]):
        preview = seq[: args.max_length]
        shown = preview[:80] + ("..." if len(preview) > 80 else "")
        print(f"  label={label}  len={len(seq)}  seq={shown}")


if __name__ == "__main__":
    main()
