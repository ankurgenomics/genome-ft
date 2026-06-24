"""Evaluate base vs. fine-tuned Nucleotide Transformer on a GenomicBenchmarks test split.

Loads the original base checkpoint (with a fresh classification head) and the
fine-tuned checkpoint, evaluates both on the held-out test split, and reports
accuracy, binary F1, and Matthews correlation coefficient (MCC) for each, plus
the deltas. Writes everything to ``eval_report.json``.

Run (after train.py):

    PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate.py \\
        --task human_enhancers_cohn \\
        --checkpoint_dir ./checkpoints_enhancers_best \\
        --max_length 500 --batch_size 16
"""

from __future__ import annotations

import argparse
import json

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import prepare_data


# Published reference numbers (test accuracy) from the GenomicBenchmarks paper
# (Gresova et al., 2023) and HyenaDNA paper (Nguyen et al., 2023). These are
# PUBLISHED REFERENCE values, NOT computed in this script. Use only as a sanity
# anchor; exact numbers vary by split, model size, and training setup.
PUBLISHED_REFERENCE = {
    "human_enhancers_cohn": {
        "metric": "accuracy",
        "genomic_benchmarks_cnn": 0.69,
        "hyenadna_reported": 0.74,
        "note": "published reference, not computed here",
    },
    "human_enhancers_ensembl": {
        "metric": "accuracy",
        "genomic_benchmarks_cnn": 0.84,
        "hyenadna_reported": 0.89,
        "note": "published reference, not computed here",
    },
    "human_nontata_promoters": {
        "metric": "accuracy",
        "genomic_benchmarks_cnn": 0.85,
        "hyenadna_reported": 0.96,
        "note": "published reference, not computed here",
    },
    "demo_coding_vs_intergenomic_seqs": {
        "metric": "accuracy",
        "genomic_benchmarks_cnn": 0.87,
        "hyenadna_reported": 0.91,
        "note": "published reference, not computed here",
    },
}


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(batch: dict) -> TensorDataset:
    if "attention_mask" in batch:
        return TensorDataset(batch["input_ids"], batch["attention_mask"], batch["labels"])
    return TensorDataset(batch["input_ids"], batch["labels"])


def unpack_batch(items, has_mask: bool, device: torch.device):
    if has_mask:
        input_ids, attention_mask, labels = items
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
    else:
        input_ids, labels = items
        inputs = {"input_ids": input_ids.to(device)}
    return inputs, labels


def extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs.get("logits", next(iter(outputs.values())))
    return outputs


@torch.no_grad()
def collect_predictions(model, loader, has_mask: bool, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []
    for items in loader:
        inputs, labels = unpack_batch(items, has_mask, device)
        logits = extract_logits(model(**inputs))
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def compute_metrics(labels, preds) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_binary": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def evaluate_model(model, loader, has_mask, device):
    labels, preds = collect_predictions(model, loader, has_mask, device)
    return compute_metrics(labels, preds), labels, preds


def main(args) -> None:
    device = select_device()
    print(f"Device: {device}")

    print(f"Loading tokenizer from fine-tuned dir: {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)

    print(f"Loading task: {args.task} (max_length={args.max_length})")
    _, _, test_batch, num_labels = prepare_data.load_task(
        args.task, args.max_length, tokenizer
    )
    has_mask = "attention_mask" in test_batch
    test_loader = DataLoader(
        make_dataset(test_batch), batch_size=args.batch_size, shuffle=False
    )

    print(f"Loading BASE model: {args.base_checkpoint}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_checkpoint, trust_remote_code=True, num_labels=num_labels
    ).to(device)
    base_metrics, _, _ = evaluate_model(base_model, test_loader, has_mask, device)

    print(f"Loading FINE-TUNED model: {args.checkpoint_dir}")
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    ).to(device)
    ft_metrics, ft_labels, ft_preds = evaluate_model(ft_model, test_loader, has_mask, device)

    delta = {k: ft_metrics[k] - base_metrics[k] for k in base_metrics}

    cm = confusion_matrix(ft_labels, ft_preds).tolist()
    report = classification_report(ft_labels, ft_preds, zero_division=0, output_dict=True)

    report = {
        "task": args.task,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "test_examples": test_batch["labels"].numel(),
        "base_checkpoint": args.base_checkpoint,
        "finetuned_checkpoint_dir": args.checkpoint_dir,
        "base_model": base_metrics,
        "finetuned_model": ft_metrics,
        "delta_finetuned_minus_base": delta,
        "confusion_matrix": cm,
        "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
        "classification_report": report,
        "published_reference": PUBLISHED_REFERENCE.get(
            args.task, {"note": "no published reference recorded for this task"}
        ),
    }

    print("\n=== Evaluation ===")
    print(f"Base      : acc={base_metrics['accuracy']:.4f}  "
          f"f1={base_metrics['f1_binary']:.4f}  mcc={base_metrics['mcc']:.4f}")
    print(f"Fine-tuned: acc={ft_metrics['accuracy']:.4f}  "
          f"f1={ft_metrics['f1_binary']:.4f}  mcc={ft_metrics['mcc']:.4f}")
    print(f"Delta     : acc={delta['accuracy']:+.4f}  "
          f"f1={delta['f1_binary']:+.4f}  mcc={delta['mcc']:+.4f}")

    # Confusion matrix for the fine-tuned model
    print("\nConfusion matrix (fine-tuned, rows=true, cols=pred):")
    print(f"            pred 0   pred 1")
    print(f"  true 0    {cm[0][0]:>6}   {cm[0][1]:>6}   (TN, FP)")
    print(f"  true 1    {cm[1][0]:>6}   {cm[1][1]:>6}   (FN, TP)")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"  precision={prec:.4f}  recall={rec:.4f}  specificity={spec:.4f}")

    ref = report["published_reference"]
    if "hyenadna_reported" in ref:
        print(f"Published reference ({ref['metric']}): "
              f"GB-CNN={ref['genomic_benchmarks_cnn']}, "
              f"HyenaDNA={ref['hyenadna_reported']} "
              f"({ref['note']})")

    with open("eval_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote eval_report.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs. fine-tuned HyenaDNA on a GenomicBenchmarks task."
    )
    parser.add_argument("--task", default="human_enhancers_cohn")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--base_checkpoint", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
