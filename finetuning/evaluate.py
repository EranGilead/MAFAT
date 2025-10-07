import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# Reuse the dataset class from the training script.
from train import HebrewQADataset  # type: ignore

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or more fine-tuned relevance models.")
    parser.add_argument(
        "--model-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing fine-tuned models (as saved by train.py).",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=SCRIPT_DIR / "data" / "training_pairs.jsonl",
        help="JSONL file with evaluation examples (query, passage, label).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=SCRIPT_DIR / "evaluation_results.csv",
        help="Where to store the aggregated metrics (CSV).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Optional cap on the number of evaluation examples (useful for smoke tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation DataLoader.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length used for tokenization (must match training to avoid truncation drift).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for evaluation: 'cuda', 'cpu', 'mps', or 'auto' to pick automatically.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for the evaluation DataLoader.",
    )
    return parser.parse_args()


def infer_device(device_arg: str | None) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - Mac GPU
        return torch.device("mps")
    return torch.device("cpu")


def maybe_slice_dataset(dataset: HebrewQADataset, max_samples: int | None, seed: int = 42) -> HebrewQADataset | Subset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[-max_samples:]
    return Subset(dataset, indices.tolist())


def compute_metrics(labels: Sequence[int], preds: Sequence[int], num_labels: int) -> dict:
    label_range = list(range(num_labels))
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=label_range,
        average=None,
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds, labels=label_range).tolist()
    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "confusion_matrix": conf,
        "precision_per_class": per_precision.tolist(),
        "recall_per_class": per_recall.tolist(),
        "f1_per_class": per_f1.tolist(),
        "support_per_class": support.tolist(),
    }


def evaluate_model(model_id: str, args: argparse.Namespace, device: torch.device) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_id)
    use_local = model_path.exists()
    load_target = model_path if use_local else model_id

    if use_local:
        LOGGER.info("Loading model from local directory: %s", model_path)
    else:
        LOGGER.info("Loading model from Hugging Face Hub: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(load_target)
    dataset = HebrewQADataset(args.data_file, tokenizer, max_length=args.max_length)
    dataset = maybe_slice_dataset(dataset, args.max_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(load_target)
    model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    progress = tqdm(dataloader, desc=f"Evaluating {model_path.name if use_local else model_id}", leave=False)
    with torch.no_grad():
        for batch in progress:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    labels_tensor = torch.cat(all_labels)
    preds_tensor = torch.cat(all_preds)

    metrics = compute_metrics(labels_tensor.numpy(), preds_tensor.numpy(), model.config.num_labels)
    model_identifier = str(model_path.resolve()) if use_local else model_id
    metrics.update(
        {
            "model_dir": model_identifier,
            "model_name": model.config._name_or_path,
            "num_samples": int(labels_tensor.numel()),
        }
    )
    return metrics


def write_results(results: Iterable[dict], output_file: Path) -> None:
    rows = list(results)
    if not rows:
        LOGGER.warning("No evaluation results to write.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_dir",
        "model_name",
        "num_samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "precision_per_class",
        "recall_per_class",
        "f1_per_class",
        "support_per_class",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["confusion_matrix"] = json.dumps(serializable["confusion_matrix"], ensure_ascii=False)
            serializable["precision_per_class"] = json.dumps(serializable["precision_per_class"], ensure_ascii=False)
            serializable["recall_per_class"] = json.dumps(serializable["recall_per_class"], ensure_ascii=False)
            serializable["f1_per_class"] = json.dumps(serializable["f1_per_class"], ensure_ascii=False)
            serializable["support_per_class"] = json.dumps(serializable["support_per_class"], ensure_ascii=False)
            writer.writerow(serializable)

    LOGGER.info("Wrote evaluation metrics for %d model(s) to %s", len(rows), output_file)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOGGER.info("Args: %s", args)

    device = infer_device(args.device)
    LOGGER.info("Using device: %s", device)

    if not args.data_file.exists():
        raise FileNotFoundError(f"Evaluation data file not found: {args.data_file}")

    results = []
    for model_id in args.model_dirs:
        try:
            metrics = evaluate_model(model_id, args, device)
        except Exception as exc:  # pragma: no cover - surface errors cleanly
            LOGGER.exception("Failed to evaluate model at %s", model_id)
            raise exc
        results.append(metrics)

    write_results(results, args.output_file)


if __name__ == "__main__":
    main()
