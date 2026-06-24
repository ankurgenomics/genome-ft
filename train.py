"""Fine-tune the Nucleotide Transformer v2 on a GenomicBenchmarks DNA classification task.

Full fine-tuning (all parameters) via AdamW + linear-warmup/cosine-decay LR schedule on
Apple Silicon MPS (or CPU). Per-epoch metrics (accuracy, F1, MCC) are written to a JSON
log; the best checkpoint by val MCC is saved automatically.

Quick smoke test:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py \\
        --task human_enhancers_cohn --limit 500 --epochs 2 --eval_limit 500

Full training run:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py \\
        --task human_enhancers_cohn --epochs 4 \\
        --lr 1e-5 --warmup_steps 300 --weight_decay 0.01 \\
        --max_grad_norm 1.0 --augment_rc --val_frac 0.15 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import prepare_data


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _copy_remote_code(model, out_dir: str) -> None:
    """Copy any custom modeling/config .py files next to the saved checkpoint.

    Models loaded with trust_remote_code rely on dynamic .py files that
    save_pretrained does not always copy. Without them the saved checkpoint
    cannot be reloaded. This copies them from the loaded module's source dir.
    """
    import inspect
    import shutil

    try:
        src_file = inspect.getsourcefile(type(model))
        if not src_file:
            return
        src_dir = os.path.dirname(src_file)
        for fname in os.listdir(src_dir):
            if fname.endswith(".py"):
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(src_dir, fname), dst)
    except Exception as exc:
        print(f"  [note] could not copy remote code files: {exc}")


def make_dataset(batch: dict) -> TensorDataset:
    """Build a TensorDataset; attention_mask is optional."""
    if "attention_mask" in batch:
        return TensorDataset(batch["input_ids"], batch["attention_mask"], batch["labels"])
    return TensorDataset(batch["input_ids"], batch["labels"])


def unpack_batch(items, has_mask: bool, device: torch.device):
    """Move a DataLoader batch to the device and return model kwargs + labels."""
    if has_mask:
        input_ids, attention_mask, labels = items
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
    else:
        input_ids, labels = items
        inputs = {"input_ids": input_ids.to(device)}
    return inputs, labels.to(device)


def extract_logits(outputs):
    """Return logits whether the HF output exposes `.logits` or is a tuple."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs.get("logits", next(iter(outputs.values())))
    return outputs


@torch.no_grad()
def evaluate_metrics(model, loader, has_mask: bool, device: torch.device) -> dict:
    """Compute accuracy, binary F1, and MCC on a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    for items in loader:
        inputs, labels = unpack_batch(items, has_mask, device)
        logits = extract_logits(model(**inputs))
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n if n else 0.0
    f1 = float(f1_score(all_labels, all_preds, average="binary", zero_division=0))
    mcc = float(matthews_corrcoef(all_labels, all_preds))
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "mcc": round(mcc, 4), "n": n}


def train(args) -> None:
    device = select_device()
    print(f"Device: {device}")

    print(f"Loading tokenizer + model from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    print(f"Loading task: {args.task} (max_length={args.max_length})")
    train_batch, val_batch, test_batch, num_labels = prepare_data.load_task(
        args.task, args.max_length, tokenizer,
        augment_rc=args.augment_rc, val_frac=args.val_frac, seed=args.seed,
    )
    if args.limit and args.limit > 0:
        import torch as _t
        g = _t.Generator().manual_seed(args.seed)
        n_tr = train_batch["labels"].numel()
        perm = _t.randperm(n_tr, generator=g)[: args.limit]
        for k in list(train_batch.keys()):
            train_batch[k] = train_batch[k][perm]
        print(f"  [subset] sampled {train_batch['labels'].numel()} shuffled train examples; "
              f"class balance {train_batch['labels'].bincount().tolist()}")

    # Pick the set used for per-epoch checkpoint selection.
    # Prefer the held-out validation split (no test leakage); fall back to test
    # only if no validation split was requested.
    has_val = val_batch["labels"].numel() > 0
    if has_val:
        eval_batch = val_batch
        eval_source = "val"
    else:
        eval_batch = test_batch
        eval_source = "test (no val split — checkpoint selection may leak)"

    if args.eval_limit and args.eval_limit > 0:
        for k in list(eval_batch.keys()):
            eval_batch[k] = eval_batch[k][: args.eval_limit]
        print(f"  [eval_limit] per-epoch eval capped at {args.eval_limit} examples")

    print(f"  num_labels={num_labels}  train={train_batch['labels'].numel()}  "
          f"val={val_batch['labels'].numel()}  test={test_batch['labels'].numel()}  "
          f"(checkpoint selection on: {eval_source})")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, trust_remote_code=True, num_labels=num_labels
    )
    model.to(device)

    has_mask = "attention_mask" in train_batch
    train_loader = DataLoader(
        make_dataset(train_batch), batch_size=args.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        make_dataset(eval_batch), batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # LR schedule: linear warmup then cosine decay to 0
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metrics_log = []
    best_mcc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        nan_detected = False
        epoch_start = time.time()
        total_batches = len(train_loader)

        for items in train_loader:
            inputs, labels = unpack_batch(items, has_mask, device)
            optimizer.zero_grad()
            logits = extract_logits(model(**inputs))
            loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                print(
                    f"  [WARNING] Non-finite loss detected at epoch {epoch} "
                    f"(batch {num_batches + 1}). Skipping update."
                )
                nan_detected = True
                break

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            num_batches += 1

            if args.log_every and num_batches % args.log_every == 0:
                elapsed = time.time() - epoch_start
                rate = num_batches / elapsed if elapsed else 0.0
                print(
                    f"  epoch {epoch} step {num_batches}/{total_batches} "
                    f"loss={loss.item():.4f} ({rate:.2f} batch/s)",
                    flush=True,
                )

        if nan_detected:
            print(f"Epoch {epoch}: aborted due to non-finite loss. Stopping training.")
            break

        avg_loss = running_loss / num_batches if num_batches else math.nan
        current_lr = scheduler.get_last_lr()[0]
        m = evaluate_metrics(model, eval_loader, has_mask, device)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
            f"val_acc={m['accuracy']:.4f}  val_f1={m['f1']:.4f}  val_mcc={m['mcc']:.4f}  "
            f"({epoch_time:.0f}s)",
            flush=True,
        )
        metrics_log.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "lr": round(current_lr, 8),
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "eval_n": m["n"],
            "epoch_time_s": round(epoch_time, 1),
        })

        # Save best checkpoint whenever val MCC improves
        if m["mcc"] > best_mcc:
            best_mcc = m["mcc"]
            best_epoch = epoch
            best_dir = args.out_dir + "_best"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            _copy_remote_code(model, best_dir)
            print(f"  [best] new best val MCC={best_mcc:.4f} at epoch {epoch} → saved to {best_dir}", flush=True)

        # Flush metrics to disk after every epoch so progress is saved if interrupted
        with open(args.metrics_out, "w") as fh:
            json.dump({"epochs": metrics_log, "config": vars(args)}, fh, indent=2)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    _copy_remote_code(model, args.out_dir)
    print(f"Saved final checkpoint (epoch {args.epochs}) to: {args.out_dir}")
    print(f"Best checkpoint (epoch {best_epoch}, val MCC={best_mcc:.4f}) saved to: {args.out_dir}_best")

    # Final TEST-set evaluation on the BEST checkpoint (the one and only time we
    # touch the test set, so the number is an honest generalisation estimate).
    if has_val:
        print("\n=== Final test-set evaluation on best checkpoint ===")
        best_model = AutoModelForSequenceClassification.from_pretrained(
            args.out_dir + "_best", trust_remote_code=True
        ).to(device)
        test_loader = DataLoader(
            make_dataset(test_batch), batch_size=args.batch_size, shuffle=False
        )
        test_m = evaluate_metrics(best_model, test_loader, has_mask, device)
        print(f"  TEST  acc={test_m['accuracy']:.4f}  f1={test_m['f1']:.4f}  "
              f"mcc={test_m['mcc']:.4f}  (n={test_m['n']})")
        # Append the test result to the metrics log
        with open(args.metrics_out, "w") as fh:
            json.dump({
                "epochs": metrics_log,
                "best_epoch": best_epoch,
                "best_val_mcc": best_mcc,
                "final_test": test_m,
                "config": vars(args),
            }, fh, indent=2)

    print(f"Metrics log: {args.metrics_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Nucleotide Transformer v2 on a GenomicBenchmarks task.")
    parser.add_argument("--task", default="human_enhancers_cohn")
    parser.add_argument("--checkpoint", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Peak learning rate (lowered from 2e-5 to reduce overfitting).")
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, cap train to this many examples (0 = use full dataset).")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of training data held out as validation for "
                             "checkpoint selection (keeps the test set unseen).")
    parser.add_argument("--log_every", type=int, default=25,
                        help="Print a progress line every N batches (0 to disable).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for subset sampling and reproducibility.")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Linear LR warmup steps before cosine decay.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (L2 regularisation).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--augment_rc", action="store_true",
                        help="Double training data with reverse complement sequences.")
    parser.add_argument("--eval_limit", type=int, default=0,
                        help="Cap eval set per epoch to this many examples for speed (0 = full).")
    parser.add_argument("--metrics_out", default="./metrics_log.json",
                        help="Path to write per-epoch metrics JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
