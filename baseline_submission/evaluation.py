import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from model import preprocess, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval + relevance pipeline using model.py preprocess/predict"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("hsrc/hsrc_train.jsonl"),
        help="HSRC dataset JSONL containing queries, paragraphs, and labels.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="Number of examples to evaluate (after shuffling with the provided seed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--use-normalization",
        action="store_true",
        help="Use Hebrew normalization when building embeddings and embedding queries.",
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help="Use BM25 lexical retrieval and rely solely on E5 embeddings.",
    )
    parser.add_argument(
        "--bm25-alpha",
        type=float,
        default=0.8,
        help="Weight for E5 similarity when mixing with BM25 (alpha * E5 + (1-alpha) * BM25).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional path to store the computed metrics as JSON.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Tuple[List[dict], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise ValueError("This evaluation currently expects an HSRC JSONL dataset")

    examples: List[dict] = []
    corpus_texts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = row.get("query", "")
            paragraphs = row.get("paragraphs", {})
            target_actions = row.get("target_actions", {})
            if not query or not paragraphs or not target_actions:
                continue
            para_keys = sorted(paragraphs.keys())
            target_keys = sorted(target_actions.keys())
            label_map: Dict[str, int] = {}
            for p_key, t_key in zip(para_keys, target_keys):
                para_data = paragraphs[p_key]
                uuid = para_data.get("uuid")
                passage = para_data.get("passage", "")
                if not uuid or not passage:
                    continue
                corpus_texts.setdefault(str(uuid), passage)
                try:
                    label = int(target_actions.get(t_key, 0))
                except (TypeError, ValueError):
                    label = 0
                label_map[str(uuid)] = label
            if not label_map:
                continue
            examples.append({"query": query, "label_map": label_map})
    return examples, corpus_texts


def compute_metrics(rankings: List[List[str]], label_maps: List[Dict[str, int]]) -> Dict[str, float]:
    if len(rankings) != len(label_maps):
        raise ValueError("Rankings and label maps must have the same length")

    hit_levels = [1, 3, 5, 10, 20]
    hits_at = {k: 0 for k in hit_levels}
    reciprocal_ranks: List[float] = []
    ndcg_scores: List[float] = []
    gold_labels: List[int] = []
    predicted_labels: List[int] = []
    missing_gold = 0

    for ranking, label_map in zip(rankings, label_maps):
        if not label_map:
            continue
        gold_id, gold_label = max(label_map.items(), key=lambda kv: (kv[1], kv[0]))
        gold_labels.append(gold_label)

        if ranking:
            predicted_id = ranking[0]
            predicted_label = label_map.get(predicted_id, 0)
        else:
            predicted_label = 0
        predicted_labels.append(predicted_label)

        if ranking and gold_id in ranking:
            rank_position = ranking.index(gold_id) + 1
            for k in hits_at:
                if rank_position <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / rank_position)
        else:
            reciprocal_ranks.append(0.0)
            missing_gold += 1

        gains = []
        for position, doc_id in enumerate(ranking[:20], start=1):
            label = label_map.get(doc_id, 0)
            gains.append((2 ** label - 1) / np.log2(position + 1))
        dcg = float(np.sum(gains)) if gains else 0.0
        ideal_labels = sorted(label_map.values(), reverse=True)[:20]
        ideal_gains = [
            (2 ** lbl - 1) / np.log2(idx + 2)
            for idx, lbl in enumerate(ideal_labels)
        ]
        idcg = float(np.sum(ideal_gains)) if ideal_gains else 0.0
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    evaluated = len(gold_labels)
    if evaluated == 0:
        raise ValueError("No evaluable examples: dataset may be empty.")

    metrics: Dict[str, float] = {f"hits@{k}": hits_at[k] / evaluated for k in hit_levels}
    metrics["mrr"] = float(np.mean(reciprocal_ranks))
    metrics["ndcg@20"] = float(np.mean(ndcg_scores))
    metrics["evaluated_examples"] = evaluated
    metrics["missing_examples"] = missing_gold

    label_set = sorted(set(gold_labels) | set(predicted_labels))
    if not label_set:
        label_set = [0]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
        labels=label_set,
        average=None,
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    metrics["accuracy"] = accuracy_score(gold_labels, predicted_labels)
    metrics["precision"] = weighted_precision
    metrics["recall"] = weighted_recall
    metrics["f1"] = weighted_f1
    metrics["precision_per_class"] = precision.tolist()
    metrics["recall_per_class"] = recall.tolist()
    metrics["f1_per_class"] = f1.tolist()
    metrics["support_per_class"] = support.tolist()
    metrics["confusion_matrix"] = confusion_matrix(
        gold_labels,
        predicted_labels,
        labels=label_set,
    ).tolist()
    metrics["label_ids"] = label_set

    return metrics


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOGGER = logging.getLogger(__name__)

    LOGGER.info("Loading HSRC dataset from %s", args.dataset_path)
    dataset, corpus_texts = load_dataset(args.dataset_path)

    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    sample_size = min(args.sample_size, len(dataset))
    sampled_examples = dataset[-sample_size:]

    if not sampled_examples:
        raise ValueError("Dataset sample is empty; adjust --sample-size or dataset path")

    relevant_ids = set()
    for example in sampled_examples:
        relevant_ids.update(example.get("label_map", {}).keys())
    corpus_dict = {pid: {"text": corpus_texts[pid]} for pid in relevant_ids if pid in corpus_texts}
    LOGGER.info("Building retrieval index over %d passages (sample-limited)", len(corpus_dict))

    if not corpus_dict:
        raise ValueError("Corpus dictionary is empty; nothing to preprocess.")

    # --- updated preprocess call ---
    preprocessed_data = preprocess(
        corpus_dict,
        use_normalization=args.use_normalization,
        use_bm25=args.use_bm25,
    )

    rankings: List[List[str]] = []
    label_maps: List[Dict[str, int]] = []

    for example in sampled_examples:
        query_text = example["query"]
        try:
            predictions = predict({"query": query_text}, preprocessed_data)
        except Exception as exc:  # pragma: no cover - surface errors cleanly
            LOGGER.exception("Prediction failed for query: %s", query_text)
            raise exc
        ranking_ids = [res.get("paragraph_uuid") for res in predictions]
        rankings.append(ranking_ids)
        label_maps.append(example.get("label_map", {}))

    metrics = compute_metrics(rankings, label_maps)
    metrics["num_samples"] = len(rankings)
    metrics["use_normalization"] = preprocessed_data.get("use_normalization", False)
    metrics["use_bm25"] = preprocessed_data.get("use_bm25", False)
    metrics["bm25_mode"] = "expander" if args.use_bm25 else "none"

    LOGGER.info("Evaluation metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.results_path:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with args.results_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Saved metrics to %s", args.results_path)


if __name__ == "__main__":
    main()
