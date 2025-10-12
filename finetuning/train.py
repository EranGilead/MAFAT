import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import torch
os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, random_split, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


class HebrewQADataset(Dataset):
    """Dataset that keeps query/passage pairs and tokenizes on demand."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Could not locate data file: {self.data_path}")

        self.examples = []
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.examples.append(
                    {
                        "query": record["query"],
                        "passage": record["passage"],
                        "label": int(record["label"]),
                    }
                )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:  # noqa: D401
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        encoding = self.tokenizer(
            example["query"],
            example["passage"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = torch.tensor(encoding["token_type_ids"], dtype=torch.long)

        return item



class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights in the loss."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights must be provided for WeightedTrainer")
        self.class_weights = class_weights.float()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None or logits is None:
            raise ValueError("Both labels and logits must be present for loss computation")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _resolve_base_dataset(dataset):
    """Return the underlying HebrewQADataset and indices for a dataset or Subset."""
    if isinstance(dataset, Subset):
        base_dataset, base_indices = _resolve_base_dataset(dataset.dataset)
        resolved = [base_indices[idx] for idx in dataset.indices]
        return base_dataset, resolved
    return dataset, list(range(len(dataset)))


def compute_label_counts(dataset) -> Counter:
    base_dataset, indices = _resolve_base_dataset(dataset)
    counter = Counter()
    for idx in indices:
        counter[base_dataset.examples[idx]["label"]] += 1
    return counter


def build_class_weights(label_counts: Counter, num_labels: int) -> torch.Tensor:
    total = sum(label_counts.values())
    weights = []
    for label in range(num_labels):
        count = label_counts.get(label, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_labels * count))
    return torch.tensor(weights, dtype=torch.float)




def log_per_class_metrics(trainer: Trainer, dataset, split_name: str, num_labels: int) -> None:
    """Run prediction on a dataset and log per-class precision/recall/F1."""
    prediction_output = trainer.predict(dataset, metric_key_prefix=split_name)
    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    labels = prediction_output.label_ids

    label_ids = list(range(num_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds, labels=label_ids)

    LOGGER.info("%s metrics:", split_name.capitalize())
    LOGGER.info("  accuracy: %.4f", accuracy)
    for idx, (p_val, r_val, f_val, supp) in enumerate(zip(precision, recall, f1, support)):
        LOGGER.info(
            "  label %d -> precision: %.4f | recall: %.4f | f1: %.4f | support: %d",
            idx,
            p_val,
            r_val,
            f_val,
            int(supp),
        )
    LOGGER.info("  confusion matrix:%s", conf)

    log_payload = {f"{split_name}/accuracy": float(accuracy)}
    for idx, (p_val, r_val, f_val, supp) in enumerate(zip(precision, recall, f1, support)):
        log_payload[f"{split_name}/precision_class_{idx}"] = float(p_val)
        log_payload[f"{split_name}/recall_class_{idx}"] = float(r_val)
        log_payload[f"{split_name}/f1_class_{idx}"] = float(f_val)
        log_payload[f"{split_name}/support_class_{idx}"] = float(supp)

    trainer.log(log_payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Hebrew relevance model.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=SCRIPT_DIR / "data" / "training_pairs.jsonl",
        help="Path to the JSONL file created by prepare_data.py.",
    )
    parser.add_argument(
        "--model-name",
        default="onlplab/alephbert-base",
        help="Base model name from Hugging Face Hub (e.g. HeBERT or AlphaBERT).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "models" / "finetuned-model",
        help="Directory where checkpoints and the final model will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--skip-training", action="store_true", help="Only save the initialized model without running training.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Linear warmup over warmup_ratio fraction of training steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps; overrides warmup_ratio when set.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay to apply.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training loss every N steps.",
    )
    parser.add_argument(
        "--evaluation-strategy",
        choices=["no", "epoch", "steps"],
        default="epoch",
        help="How often to run evaluation.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Evaluation interval when using the 'steps' strategy.",
    )
    parser.add_argument(
        "--save-strategy",
        choices=["no", "epoch", "steps"],
        default=None,
        help="Checkpoint saving strategy; defaults to evaluation strategy if unset.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Checkpoint interval when using the 'steps' strategy.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep on disk (None keeps all).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset reserved for validation (0 disables validation).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and weight initialization.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=5,
        help="Number of relevance labels in the dataset.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training examples for quick experiments.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit evaluation examples for quick experiments.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (if supported by the hardware).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 mixed precision (for Ampere+ GPUs).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        default="linear",
        help="Learning-rate schedule (Transformer Trainer options).",
    )

    return parser.parse_args()


def build_compute_metrics(num_labels: int):
    label_ids = list(range(num_labels))

    def compute_metrics(pred):
        logits = pred.predictions
        preds = np.argmax(logits, axis=-1)
        labels = pred.label_ids

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )
        per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
            labels,
            preds,
            labels=label_ids,
            average=None,
            zero_division=0,
        )
        accuracy = accuracy_score(labels, preds)
        conf = confusion_matrix(labels, preds, labels=label_ids)

        LOGGER.info("Eval confusion matrix:%s", conf)

        metrics = {
            "accuracy": accuracy,
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        }
        for idx, (p_val, r_val, f_val, supp) in enumerate(zip(per_precision, per_recall, per_f1, support)):
            metrics[f"precision_class_{idx}"] = float(p_val)
            metrics[f"recall_class_{idx}"] = float(r_val)
            metrics[f"f1_class_{idx}"] = float(f_val)
            metrics[f"support_class_{idx}"] = float(supp)
        return metrics

    return compute_metrics


def maybe_slice_dataset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:max_samples]
    return Subset(dataset, indices.tolist())


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOGGER.info("Args: %s", args)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    train_file = args.train_file.resolve()
    output_dir = args.output_dir.resolve()
    LOGGER.info("Loading dataset from %s", train_file)
    dataset = HebrewQADataset(train_file, tokenizer, max_length=args.max_length)
    if len(dataset) == 0:
        raise ValueError("The training dataset is empty. Run prepare_data.py first or check the data path.")

    has_validation = args.val_ratio > 0 and len(dataset) > 1
    if has_validation:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, eval_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, eval_dataset = dataset, None

    train_dataset = maybe_slice_dataset(train_dataset, args.max_train_samples, args.seed)
    if eval_dataset is not None:
        eval_dataset = maybe_slice_dataset(eval_dataset, args.max_eval_samples, args.seed)

    label_counts = compute_label_counts(train_dataset)
    LOGGER.info("Training label distribution after sampling: %s", label_counts)
    class_weights = build_class_weights(label_counts, args.num_labels)
    LOGGER.info("Computed class weights: %s", class_weights.tolist())

    eval_strategy = args.evaluation_strategy
    if eval_dataset is None and eval_strategy != "no":
        LOGGER.warning("No validation split detected; falling back to 'no' evaluation strategy.")
        eval_strategy = "no"

    save_strategy = args.save_strategy or eval_strategy
    logging_strategy = "steps" if args.logging_steps > 0 else "epoch"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_strategy=logging_strategy,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_strategy != "no",
        metric_for_best_model="f1",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=["tensorboard"],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(args.num_labels) if eval_strategy != "no" else None,
        class_weights=class_weights,
    )

    if not args.skip_training:
        trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    else:
        LOGGER.info("skip-training enabled; saving initialized model without updates")

    LOGGER.info("Collecting per-class metrics on training split...")
    log_per_class_metrics(trainer, train_dataset, "train", args.num_labels)
    if eval_dataset is not None:
        LOGGER.info("Collecting per-class metrics on evaluation split...")
        log_per_class_metrics(trainer, eval_dataset, "eval", args.num_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()

