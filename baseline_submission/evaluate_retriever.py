import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from model import E5Retriever, simple_tokenize
from normalization import normalize_text, load_hebrew_nlp

LOGGER = logging.getLogger(__name__)


def load_passages(path: Path) -> Dict[str, str]:
    passages: Dict[str, str] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            paragraphs = row.get('paragraphs', {})
            for para in paragraphs.values():
                uuid = para.get('uuid')
                passage = para.get('passage', "")
                if uuid and passage and uuid not in passages:
                    passages[str(uuid)] = passage
    return passages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the retriever stage using precomputed passage embeddings."
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("paragraph_embeddings_e5.pkl"),
        help="Pickle file mapping paragraph UUID to its E5 embedding.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("hsrc/hsrc_train.jsonl"),
        help="HSRC dataset JSONL with queries, paragraphs, and target_actions.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional Hugging Face model id/path for the E5 retriever (defaults to local cache).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="Number of query examples to evaluate (after shuffling).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of passages considered a hit for recall@K.",
    )
    parser.add_argument(
        "--embedding-topk",
        type=int,
        default=100,
        help="Number of embedding-based candidates to take before union (ignored in BM25-only mode).",
    )
    parser.add_argument(
        "--bm25-topk",
        type=int,
        default=0,
        help="Number of BM25 candidates to take and union with embeddings (set >0 to enable BM25 expansion).",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable Hebrew normalization when embedding queries.",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Evaluate using BM25 only (skip E5 embeddings).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for the retriever (e.g., 'cuda', 'mps').",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional path to store the metrics as JSON.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    LOGGER.info("Loading passage embeddings from %s", path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Embeddings file must be a mapping of paragraph_uuid -> vector")
    paragraph_ids = list(data.keys())
    matrix = np.vstack([np.asarray(data[pid], dtype=np.float32) for pid in paragraph_ids])
    LOGGER.info("Loaded %d embeddings with dimension %d", matrix.shape[0], matrix.shape[1])
    return paragraph_ids, matrix


def load_dataset(path: Path) -> List[dict]:
    LOGGER.info("Loading HSRC dataset from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = record.get("query", "")
            paragraphs = record.get("paragraphs", {})
            target_actions = record.get("target_actions", {})
            if not query or not paragraphs or not target_actions:
                continue
            label_map: Dict[str, int] = {}
            para_keys = sorted(paragraphs.keys())
            target_keys = sorted(target_actions.keys())
            for p_key, t_key in zip(para_keys, target_keys):
                uuid = paragraphs[p_key].get("uuid")
                if not uuid:
                    continue
                label = target_actions.get(t_key)
                if label is None:
                    continue
                try:
                    label_map[str(uuid)] = int(label)
                except ValueError:
                    continue
            if not label_map:
                continue
            examples.append({
                "query": query,
                "label_map": label_map,
            })
    LOGGER.info("Loaded %d query examples", len(examples))
    return examples


def compute_recall(
    rankings: List[List[str]],
    label_maps: List[Dict[str, int]],
    paragraph_ids: List[str],
    top_k: int,
) -> Dict[str, float]:
    corpus_id_set = set(paragraph_ids)

    total = 0
    hits = 0
    avg_hits = 0.0
    missing_relevant = 0
    queries_with_available_relevant = 0

    hits_by_label: Dict[int, int] = {label: 0 for label in range(5)}
    missed_by_label: Dict[int, int] = {label: 0 for label in range(5)}

    for ranking, label_map in zip(rankings, label_maps):
        relevant_ids = {pid for pid, lbl in label_map.items() if lbl > 0}
        available_relevant = relevant_ids & corpus_id_set
        if not available_relevant:
            continue
        queries_with_available_relevant += 1

        total += 1
        observed = set(ranking[:top_k])
        overlap = observed & available_relevant
        if overlap:
            hits += 1
        else:
            missing_relevant += 1
        avg_hits += len(overlap)

        for pid in available_relevant:
            label = label_map.get(pid, 0)
            if pid in overlap:
                hits_by_label[label] = hits_by_label.get(label, 0) + 1
            else:
                missed_by_label[label] = missed_by_label.get(label, 0) + 1

    metrics = {
        "queries_with_relevant_passages": total,
        "hits@{}".format(top_k): hits,
        "recall@{}".format(top_k): hits / total if total else 0.0,
        "avg_hits_within_top{}".format(top_k): avg_hits / total if total else 0.0,
        "queries_without_hit@{}".format(top_k): missing_relevant,
        "queries_total": len(label_maps),
        "queries_without_relevant_in_corpus": len(label_maps) - queries_with_available_relevant,
    }

    total_hits_all = sum(hits_by_label.values())
    total_relevant_all = total_hits_all + sum(missed_by_label.values())
    metrics["recall_overall"] = total_hits_all / total_relevant_all if total_relevant_all else 0.0

    recall_by_label = {}
    for label in sorted(set(list(hits_by_label.keys()) + list(missed_by_label.keys()))):
        if label == 0:
            continue
        hits_label = hits_by_label.get(label, 0)
        total_label = hits_label + missed_by_label.get(label, 0)
        if total_label > 0:
            recall_by_label[label] = hits_label / total_label
    metrics["recall_by_label"] = recall_by_label
    metrics["hits_by_label"] = hits_by_label
    metrics["missed_by_label"] = missed_by_label
    return metrics


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset = load_dataset(args.dataset_path)
    corpus_texts = load_passages(args.dataset_path)

    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    sample_size = min(args.sample_size, len(dataset))
    sampled = dataset[-sample_size:]
    LOGGER.info("Evaluating %d sampled queries", len(sampled))

    normalization_pipeline = None
    if not args.no_normalization:
        try:
            normalization_pipeline = load_hebrew_nlp()
            LOGGER.info("Loaded hebspacy pipeline for query normalization.")
        except ImportError:
            LOGGER.warning("hebspacy not available; proceeding without query normalization.")
            normalization_pipeline = None

    use_embeddings = not args.bm25_only

    embedding_topk = args.embedding_topk if use_embeddings else 0
    bm25_topk = args.bm25_topk

    if use_embeddings:
        paragraph_ids, paragraph_matrix = load_embeddings(args.embeddings_path)
        embedding_topk = min(embedding_topk, len(paragraph_ids))
    else:
        paragraph_ids = sorted(corpus_texts.keys())
        paragraph_matrix = None
        embedding_topk = 0

    bm25_required = args.bm25_only or bm25_topk > 0
    bm25 = None
    if bm25_required:
        passages = [corpus_texts[pid] for pid in paragraph_ids]
        if normalization_pipeline is not None:
            normalized_passages = []
            for passage in passages:
                try:
                    normalized_passages.append(normalize_text(passage, nlp=normalization_pipeline))
                except Exception as norm_err:
                    LOGGER.warning("Normalization failed for passage: %s", norm_err)
                    normalized_passages.append(passage)
        else:
            normalized_passages = passages
        tokenized_corpus = [simple_tokenize(passage) for passage in normalized_passages]
        bm25 = BM25Okapi(tokenized_corpus)
        if args.bm25_only and bm25_topk == 0:
            bm25_topk = args.top_k
    if args.bm25_only:
        use_embeddings = False
        embedding_topk = 0

    queries = []
    label_maps = []
    normalized_queries = []
    for example in sampled:
        q = example["query"]
        queries.append(q)
        label_maps.append(example["label_map"])
        if normalization_pipeline is not None:
            try:
                normalized_queries.append(normalize_text(q, nlp=normalization_pipeline))
            except Exception as norm_err:
                LOGGER.warning("Normalization failed for query: %s", norm_err)
                normalized_queries.append(q)
        else:
            normalized_queries.append(q)

    rankings: List[List[str]] = []
    top_k = args.top_k if args.top_k > 0 else len(paragraph_ids)

    embedding_scores = None
    if use_embeddings:
        LOGGER.info("Embedding queries with E5 retriever...")
        retriever = E5Retriever(model_name=args.model_name, device=args.device)
        embedding_scores = retriever.embed_texts(normalized_queries, is_query=True, batch_size=8) @ paragraph_matrix.T

    if bm25_required and bm25 is None:
        raise ValueError("BM25 index was not built; cannot use BM25 results.")

    for i, normalized_query in enumerate(normalized_queries):
        combined_indices: List[int] = []
        seen = set()

        if use_embeddings and embedding_scores is not None:
            row = embedding_scores[i]
            k_emb = min(embedding_topk, len(row))
            if top_k:
                k_emb = min(k_emb, top_k)
            if k_emb > 0:
                if k_emb >= len(row):
                    embed_indices = np.argsort(row)[::-1][:k_emb]
                else:
                    embed_indices = np.argpartition(row, -k_emb)[-k_emb:]
                    embed_indices = embed_indices[np.argsort(row[embed_indices])[::-1]]
                for idx in embed_indices:
                    if idx not in seen:
                        combined_indices.append(idx)
                        seen.add(idx)

        if bm25 is not None:
            k_bm25 = bm25_topk
            if args.bm25_only and k_bm25 == 0:
                k_bm25 = top_k
            if k_bm25 == 0 and not args.bm25_only:
                k_bm25 = min(top_k, len(paragraph_ids)) if top_k else len(paragraph_ids)
            k_bm25 = min(k_bm25, len(paragraph_ids))
            if k_bm25 > 0:
                tokens = simple_tokenize(normalized_query)
                scores = bm25.get_scores(tokens)
                if k_bm25 >= len(scores):
                    bm25_indices = np.argsort(scores)[::-1][:k_bm25]
                else:
                    bm25_indices = np.argpartition(scores, -k_bm25)[-k_bm25:]
                    bm25_indices = bm25_indices[np.argsort(scores[bm25_indices])[::-1]]
                for idx in bm25_indices:
                    if idx not in seen:
                        combined_indices.append(idx)
                        seen.add(idx)

        k_used = top_k if top_k else len(combined_indices)
        if k_used:
            combined_indices = combined_indices[:k_used]

        rankings.append([paragraph_ids[idx] for idx in combined_indices])

    metrics = compute_recall(rankings, label_maps, paragraph_ids, top_k)
    metrics["bm25_only"] = args.bm25_only
    metrics["use_embeddings"] = use_embeddings
    metrics["use_normalization"] = normalization_pipeline is not None and not args.no_normalization
    metrics["embedding_topk"] = embedding_topk if use_embeddings else 0
    metrics["bm25_topk"] = bm25_topk if bm25 is not None else 0
    metrics["top_k"] = top_k
    metrics["sampled_queries"] = len(sampled)
    LOGGER.info("Retriever metrics: %s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.results_path:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with args.results_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Saved metrics to %s", args.results_path)


if __name__ == "__main__":
    main()
