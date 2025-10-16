
import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from model import E5Retriever, BGEReranker
from normalization import normalize_text, load_hebrew_nlp

LOGGER = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality using precomputed paragraph embeddings and a query encoder. "
            "Samples HSRC queries and evaluates retrieval + relevance metrics."
        )
    )
    parser.add_argument(
        "--embeddings-path",
        required=True,
        type=Path,
        help="Path to a pickle file that maps paragraph IDs to embedding vectors.",
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
        "--device",
        default=None,
        help="Device to run inference on (e.g. 'cuda', 'cuda:0', 'mps', 'cpu'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--retriever-model",
        default=None,
        help="Model identifier or path for the retriever (E5). Defaults to local cache if omitted.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Model identifier or path for the reranker (BGE). Defaults to local cache if omitted.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=100,
        help="Number of passages retrieved before reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=100,
        help="Number of passages to keep after reranking for metric computation.",
    )
    parser.add_argument(
        "--normalize-queries",
        action="store_true",
        help="Apply Hebrew normalization (needs hebspacy) before retrieval.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional path to store the computed metrics as JSON.",
    )
    return parser.parse_args()


def infer_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected embeddings pickle to contain a dict, got {type(data)}")
    paragraph_ids = list(data.keys())
    try:
        matrix = np.vstack([np.asarray(data[pid], dtype=np.float32) for pid in paragraph_ids])
    except Exception as exc:  # pragma: no cover - defensively handle malformed files
        raise ValueError("Failed to stack embeddings into a matrix") from exc
    return paragraph_ids, matrix




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



def retrieve_and_rerank(
    retriever: E5Retriever,
    reranker: BGEReranker,
    paragraph_ids: List[str],
    paragraph_matrix: np.ndarray,
    corpus_texts: Dict[str, str],
    query_text: str,
    retrieval_top_k: int,
    rerank_top_k: int,
) -> List[str]:
    query_vec = retriever.embed_texts([query_text], is_query=True, batch_size=1)[0]
    scores = paragraph_matrix @ query_vec
    candidate_indices = np.argsort(-scores)

    candidate_ids: List[str] = []
    candidate_passages: List[str] = []
    for idx in candidate_indices:
        pid = paragraph_ids[idx]
        passage = corpus_texts.get(pid)
        if not passage:
            continue
        candidate_ids.append(pid)
        candidate_passages.append(passage)
        if len(candidate_ids) >= retrieval_top_k:
            break

    if not candidate_ids:
        return []

    rerank_k = min(len(candidate_ids), rerank_top_k)
    reranked = reranker.rerank(
        query_text,
        candidate_passages,
        candidate_ids,
        top_k=rerank_k,
    )
    return [pid for pid, _ in reranked]



def compute_metrics(
    rankings: List[List[str]],
    label_maps: List[Dict[str, int]],
) -> Dict[str, float]:
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
    LOGGER.info("Loading paragraph embeddings from %s", args.embeddings_path)
    paragraph_ids, paragraph_matrix = load_embeddings(args.embeddings_path)
    LOGGER.info("Loaded %d paragraph embeddings (dim=%d)", len(paragraph_ids), paragraph_matrix.shape[1])

    LOGGER.info("Loading HSRC dataset from %s", args.dataset_path)
    dataset, corpus_texts = load_dataset(args.dataset_path)
    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    sample_size = min(args.sample_size, len(dataset))
    sampled_examples = dataset[-sample_size:]

    filtered_examples: List[dict] = []
    for item in sampled_examples:
        query = item.get("query")
        label_map = item.get("label_map")
        if not isinstance(query, str) or not query.strip() or not isinstance(label_map, dict):
            continue
        if not label_map:
            continue
        filtered_examples.append({"query": query, "label_map": label_map})

    if not filtered_examples:
        raise ValueError("No valid examples after filtering for queries with label mappings.")

    device = infer_device(args.device)
    if args.normalize_queries:
        try:
            pipeline = load_hebrew_nlp()
            LOGGER.info("Loaded hebspacy pipeline for query normalization.")
        except ImportError:
            LOGGER.warning("hebspacy not available; proceeding without normalization.")
            args.normalize_queries = False
            pipeline = None
        else:
            for example in filtered_examples:
                example["normalized_query"] = normalize_text(
                    example["query"], nlp=pipeline, lemmatize=True
                )
    else:
        pipeline = None

    retriever = E5Retriever(model_name=args.retriever_model, device=str(device))
    reranker = BGEReranker(model_name=args.reranker_model, device=str(device))

    retriever.corpus_ids = paragraph_ids
    retriever.corpus_embeddings = paragraph_matrix

    LOGGER.info("Evaluating %d queries (top_k=%d, rerank_k=%d)", len(filtered_examples), args.retrieval_top_k, args.rerank_top_k)

    rankings: List[List[str]] = []
    label_maps = []
    for example in filtered_examples:
        query_text = example["normalized_query"] if args.normalize_queries else example["query"]
        ranking_ids = retrieve_and_rerank(
            retriever,
            reranker,
            paragraph_ids,
            paragraph_matrix,
            corpus_texts,
            query_text,
            args.retrieval_top_k,
            args.rerank_top_k,
        )
        rankings.append(ranking_ids)
        label_maps.append(example["label_map"])

    metrics = compute_metrics(rankings, label_maps)
    metrics["num_samples"] = len(rankings)
    metrics["retrieval_top_k"] = args.retrieval_top_k
    metrics["rerank_top_k"] = min(args.rerank_top_k, args.retrieval_top_k)

    LOGGER.info("Evaluation metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.results_path:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with args.results_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Saved metrics to %s", args.results_path)

if __name__ == "__main__":
    main()
