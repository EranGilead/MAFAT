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
        type=Path,
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
        default=None,
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
        help="Device to run evaluation on (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for the evaluation DataLoader.",
    )
    return parser.parse_args()


def infer_device(device_arg: str | None) -> torch.device:
    if device_arg:
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds, labels=label_range).tolist()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf,
    }


def evaluate_model(model_dir: Path, args: argparse.Namespace, device: torch.device) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = HebrewQADataset(args.data_file, tokenizer, max_length=args.max_length)
    dataset = maybe_slice_dataset(dataset, args.max_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    progress = tqdm(dataloader, desc=f"Evaluating {model_dir.name}", leave=False)
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
    metrics.update(
        {
            "model_dir": str(model_dir.resolve()),
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
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["confusion_matrix"] = json.dumps(serializable["confusion_matrix"], ensure_ascii=False)
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
    for model_dir in args.model_dirs:
        try:
            metrics = evaluate_model(model_dir, args, device)
        except Exception as exc:  # pragma: no cover - surface errors cleanly
            LOGGER.exception("Failed to evaluate model at %s", model_dir)
            raise exc
        results.append(metrics)

    write_results(results, args.output_file)


if __name__ == "__main__":
    main()
